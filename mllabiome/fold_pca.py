"""Fold-local questionnaire PCA targets for leakage-controlled CV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FoldPCATargetSpec:
    """Describe questionnaire columns and fixed reference domains for fold PCA."""

    questionnaire_columns: tuple[str, ...]
    reference_loadings_path: Path | str
    domain_to_component: tuple[tuple[str, str], ...]
    n_components: int = 4
    promax_power: int = 4
    min_alignment_congruence: float = 0.85
    reference_n: int | None = None

    def __post_init__(self) -> None:
        if len(self.questionnaire_columns) < self.n_components:
            raise ValueError("Fold PCA needs at least as many questionnaire columns as components.")
        if len(set(self.questionnaire_columns)) != len(self.questionnaire_columns):
            raise ValueError("Fold PCA questionnaire columns must be unique.")
        if len(self.domain_to_component) != self.n_components:
            raise ValueError("Fold PCA must map each component to exactly one domain.")
        domain_names = [str(domain) for domain, _ in self.domain_to_component]
        component_names = [str(component) for _, component in self.domain_to_component]
        if len(set(domain_names)) != len(domain_names):
            raise ValueError("Fold PCA domain names must be unique.")
        if len(set(component_names)) != len(component_names):
            raise ValueError("Fold PCA reference component names must be unique.")
        if not 0.0 <= float(self.min_alignment_congruence) <= 1.0:
            raise ValueError("min_alignment_congruence must be between 0 and 1.")
        if self.reference_n is not None and int(self.reference_n) <= 0:
            raise ValueError("reference_n must be a positive integer when provided.")

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(str(domain) for domain, _ in self.domain_to_component)

    @property
    def reference_components(self) -> tuple[str, ...]:
        return tuple(str(component) for _, component in self.domain_to_component)

    def domain_index(self, domain: str) -> int:
        try:
            return self.domains.index(str(domain))
        except ValueError as exc:
            raise ValueError(f"Unknown fold PCA domain {domain!r}; expected one of {self.domains!r}.") from exc


@dataclass(frozen=True)
class FoldPCASplitTargets:
    train_scores: np.ndarray
    eval_scores: np.ndarray
    diagnostics: Mapping[str, float | int | str]


def load_reference_loadings(
    path: Path | str,
    questionnaire_columns: Sequence[str],
    component_names: Sequence[str],
) -> np.ndarray:
    """Load and order a CSV loading matrix using its first column as row labels."""
    reference_path = Path(path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference PCA loadings file not found: {reference_path}")
    frame = pd.read_csv(reference_path, index_col=0)
    missing_rows = [name for name in questionnaire_columns if name not in frame.index]
    missing_columns = [name for name in component_names if name not in frame.columns]
    if missing_rows or missing_columns:
        raise ValueError(
            "Reference PCA loading matrix is missing expected labels: "
            f"rows={missing_rows!r}, columns={missing_columns!r}."
        )
    ordered = frame.loc[list(questionnaire_columns), list(component_names)]
    values = ordered.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.shape != (len(questionnaire_columns), len(component_names)):
        raise ValueError("Reference PCA loading matrix has an unexpected shape.")
    if not np.isfinite(values).all():
        raise ValueError("Reference PCA loadings contain missing or non-finite values.")
    return values


def _varimax(loadings: np.ndarray, *, tolerance: float = 1e-5, max_iter: int = 1000):
    """Kaiser-normalized Varimax rotation, matching psych::Promax initialization."""
    row_norm = np.sqrt(np.sum(loadings**2, axis=1))
    row_norm[row_norm == 0.0] = 1.0
    normalized = loadings / row_norm[:, None]
    n_rows, n_components = normalized.shape
    rotation = np.eye(n_components)
    previous = 0.0
    for _ in range(max_iter):
        rotated = normalized @ rotation
        u, singular, vt = np.linalg.svd(
            normalized.T
            @ (rotated**3 - rotated @ np.diag(np.sum(rotated**2, axis=0)) / n_rows),
            full_matrices=False,
        )
        rotation = u @ vt
        objective = float(np.sum(singular))
        if previous > 0.0 and objective - previous < tolerance * previous:
            break
        previous = objective
    return (normalized @ rotation) * row_norm[:, None], rotation


def _promax(loadings: np.ndarray, power: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promax pattern loadings, factor correlations, and structure loadings."""
    varimax_loadings, varimax_rotation = _varimax(loadings)
    target = varimax_loadings * np.abs(varimax_loadings) ** (int(power) - 1)
    # psych::Promax uses the least-squares target rotation and scales each
    # transform column from diag((U' U)^-1); whitening U would incorrectly
    # make the result orthogonal and remove the Promax factor correlations.
    transform = np.linalg.lstsq(varimax_loadings, target, rcond=None)[0]
    try:
        diagonal = np.diag(np.linalg.inv(transform.T @ transform))
    except np.linalg.LinAlgError:
        diagonal = np.diag(np.linalg.pinv(transform.T @ transform))
    diagonal = np.maximum(diagonal, np.finfo(float).eps)
    transform = transform @ np.diag(np.sqrt(diagonal))
    pattern = varimax_loadings @ transform
    inverse = np.linalg.inv(transform)
    phi = inverse @ inverse.T
    structure = pattern @ phi
    return pattern, phi, structure


def fit_fold_pca_targets(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    reference_loadings: np.ndarray,
    spec: FoldPCATargetSpec,
) -> FoldPCASplitTargets:
    """Fit standardized four-component Promax PCA on train rows and score both sets.

    The scoring calculation follows psych::principal's correlation-matrix PCA and
    regression (Thurstone) score weights. Component order and signs are then
    aligned to the independent reference loading matrix.
    """
    train = np.asarray(train_values, dtype=float)
    evaluate = np.asarray(eval_values, dtype=float)
    reference = np.asarray(reference_loadings, dtype=float)
    n_features = len(spec.questionnaire_columns)
    n_components = int(spec.n_components)
    if train.ndim != 2 or train.shape[1] != n_features:
        raise ValueError(f"Fold PCA training matrix must have {n_features} columns.")
    if evaluate.ndim != 2 or evaluate.shape[1] != n_features:
        raise ValueError(f"Fold PCA evaluation matrix must have {n_features} columns.")
    if reference.shape != (n_features, n_components):
        raise ValueError("Reference loading matrix does not match the PCA specification.")
    if len(train) <= n_components or len(evaluate) == 0:
        raise ValueError("Fold PCA needs more training rows than components and non-empty evaluation rows.")
    if not np.isfinite(train).all() or not np.isfinite(evaluate).all():
        raise ValueError("Questionnaire values for fold PCA must be complete and finite.")

    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=1)
    constant = np.flatnonzero(~np.isfinite(scale) | (scale <= np.finfo(float).eps))
    if len(constant):
        names = [spec.questionnaire_columns[int(i)] for i in constant]
        raise ValueError(f"Fold PCA has constant questionnaire columns: {names!r}.")
    z_train = (train - mean) / scale
    z_eval = (evaluate - mean) / scale
    correlation = np.corrcoef(z_train, rowvar=False)
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues = np.maximum(eigenvalues[order], np.finfo(float).eps)
    eigenvectors = eigenvectors[:, order]
    loadings = eigenvectors * np.sqrt(eigenvalues)[None, :]
    sign = np.sign(loadings.sum(axis=0))
    sign[sign == 0.0] = 1.0
    loadings *= sign[None, :]

    pattern, phi, structure = _promax(loadings, spec.promax_power)
    # psych::principal orders rotated components by descending SS loading and
    # fixes the arbitrary sign using the column sum before calculating scores.
    ss = np.sum(pattern**2, axis=0)
    component_order = np.argsort(ss)[::-1]
    pattern = pattern[:, component_order]
    structure = structure[:, component_order]
    phi = phi[np.ix_(component_order, component_order)]
    sign = np.sign(pattern.sum(axis=0))
    sign[sign == 0.0] = 1.0
    pattern *= sign[None, :]
    structure *= sign[None, :]
    phi = sign[:, None] * phi * sign[None, :]

    # Regression scoring weights: inverse training correlation x structure.
    weights = np.linalg.solve(correlation, structure)
    train_scores = z_train @ weights
    eval_scores = z_eval @ weights

    denom = np.linalg.norm(pattern, axis=0)[:, None] * np.linalg.norm(reference, axis=0)[None, :]
    congruence = (pattern.T @ reference) / np.maximum(denom, np.finfo(float).eps)
    fold_components, reference_components = linear_sum_assignment(-np.abs(congruence))
    permutation = np.empty(n_components, dtype=int)
    signs = np.ones(n_components, dtype=float)
    aligned_congruence = np.empty(n_components, dtype=float)
    for fold_component, reference_component in zip(fold_components, reference_components):
        permutation[reference_component] = fold_component
        value = float(congruence[fold_component, reference_component])
        signs[reference_component] = -1.0 if value < 0.0 else 1.0
        aligned_congruence[reference_component] = abs(value)
    train_scores = train_scores[:, permutation] * signs[None, :]
    eval_scores = eval_scores[:, permutation] * signs[None, :]

    diagnostics: dict[str, float | int | str] = {
        "pca_train_n": int(len(train)),
        "pca_mean_abs_congruence": float(np.mean(aligned_congruence)),
        "pca_min_abs_congruence": float(np.min(aligned_congruence)),
        "pca_alignment_below_threshold": int(
            np.min(aligned_congruence) < float(spec.min_alignment_congruence)
        ),
    }
    for index, component_name in enumerate(spec.reference_components):
        diagnostics[f"pca_congruence_{component_name}"] = float(aligned_congruence[index])
        diagnostics[f"pca_matched_fold_component_{component_name}"] = int(permutation[index] + 1)
        diagnostics[f"pca_sign_applied_{component_name}"] = int(signs[index] < 0.0)
    return FoldPCASplitTargets(train_scores, eval_scores, diagnostics)

