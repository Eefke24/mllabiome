from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from . import report as _report
from ._evaluation_protocols import is_lodo_protocol
from .configs_sweep import Sweep, _normalise_sweep_task, _target_task, target_sweeps
from .console import info, path_table, phase_progress, stage, success
from .final_models import build_final_models
from .metrics import compute_regression_metrics, metric_is_loss
from .regression_explainability import _write_regression_explainability_figures
from .report_compute import compute_display, run_compute_accounting
from .report_oof import _mpma_b_composition, oof_section_html, write_report
from .report_statistics import (
    _corrected_resampled_t_test,
    _exact_sign_flip_test,
    _paired_bootstrap_difference,
)
from .storage import read_table, table_exists, write_table
from .utils import (
    REGRESSION_METRIC_COLUMNS,
    dump_json_standard,
    prepare_report_html_path,
)

_REGRESSION_DISPLAY_METRICS = (
    "RMSE",
    "MAE",
    "R2",
    "PearsonR",
    "SpearmanR",
)


def _read_table(path: Path) -> pd.DataFrame:

    try:
        return read_table(path)

    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:

    if not path.exists() or path.stat().st_size == 0:
        return {}

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))

    except Exception:
        return {}

    return obj if isinstance(obj, dict) else {}


def _finite(values: pd.Series | np.ndarray | list[Any]) -> np.ndarray:

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)

    return arr[np.isfinite(arr)]


def _bootstrap_mean(
    values: np.ndarray, seed: int, n_bootstrap: int = 2000
) -> tuple[float, float]:

    arr = np.asarray(values, dtype=float)

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return float("nan"), float("nan")

    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(int(seed))

    draws = rng.choice(arr, size=(int(n_bootstrap), len(arr)), replace=True).mean(
        axis=1
    )

    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _holm_adjust(values: list[float]) -> list[float]:

    if not values:
        return []

    p = np.asarray(values, dtype=float)

    order = np.argsort(p)

    out = np.full(len(p), np.nan, dtype=float)

    running = 0.0

    m = len(p)

    for rank, index in enumerate(order):
        adjusted = min(1.0, float(p[index]) * float(m - rank))

        running = max(running, adjusted)

        out[index] = running

    return [float(x) for x in out]


def _regression_strategy_frames(root: Path) -> dict[str, pd.DataFrame]:

    out: dict[str, pd.DataFrame] = {}

    mpma_b = _read_table(root / "results" / "mpma_b_outer_results.parquet")

    if not mpma_b.empty:
        out["MPMA-B"] = mpma_b

    mpma_e = _read_table(root / "ensembling" / "mpma_e_outer_results.parquet")

    if not mpma_e.empty:
        out["MPMA-E"] = mpma_e

    return out


def _regression_prediction_frames(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "MPMA-B": root / "predictions" / "mpma_b_outer_predictions.parquet",
        "MPMA-E": root / "ensembling" / "ensemble_predictions.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for strategy, path in paths.items():
        frame = _read_table(path)
        if not frame.empty:
            out[strategy] = frame
    return out


def _regression_repeat_id(value: Any) -> str:
    text = str(value)
    if text.startswith("r") and "_o" in text:
        return text.split("_o", 1)[0]
    return "r0"


def _prepare_regression_oof(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = frame.copy()
    if "outer_split_key" not in out.columns and "split_key" in out.columns:
        out["outer_split_key"] = out["split_key"]
    required = {"outer_split_key", "sample_id", "y_true", "y_pred"}
    if out.empty or not required.issubset(out.columns):
        return pd.DataFrame()
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str)
    if "subject_id" in out.columns:
        if out["subject_id"].isna().any():
            raise ValueError(
                "Held-out regression predictions contain missing subject_id values."
            )
        out["_subject_id"] = out["subject_id"].astype(str)
    else:
        out["_subject_id"] = out["sample_id"]
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    valid = np.isfinite(out["y_true"].to_numpy(dtype=float)) & np.isfinite(
        out["y_pred"].to_numpy(dtype=float)
    )
    out = out.loc[valid].copy()
    key = str(protocol).strip().lower()
    if is_lodo_protocol(key):
        out["_cluster"] = out["outer_split_key"]
        out["_repeat"] = "r0"
        subject_cohorts = out.groupby("_subject_id", sort=False)["_cluster"].nunique()
        if bool((subject_cohorts > 1).any()):
            raise ValueError(
                "A subject_id occurs in more than one held-out regression LODO cohort."
            )
        duplicate_keys = ["_cluster", "sample_id"]
    else:
        out["_repeat"] = out["outer_split_key"].map(_regression_repeat_id)
        out["_cluster"] = out["_repeat"]
        duplicate_keys = ["_repeat", "sample_id"]
    if out.duplicated(duplicate_keys).any():
        raise ValueError(
            "Held-out regression predictions contain duplicate inference rows."
        )
    return out.reset_index(drop=True)


def _regression_mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in REGRESSION_METRIC_COLUMNS:
        arr = np.asarray([row.get(metric, np.nan) for row in values], dtype=float)
        arr = arr[np.isfinite(arr)]
        result[metric] = float(np.mean(arr)) if len(arr) else float("nan")
    return result


def _regression_oof_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {metric: float("nan") for metric in REGRESSION_METRIC_COLUMNS}
    return compute_regression_metrics(
        frame["y_true"].to_numpy(dtype=float), frame["y_pred"].to_numpy(dtype=float)
    )


def _regression_point_estimands(
    frame: pd.DataFrame, protocol: str
) -> dict[str, dict[str, float]]:
    key = str(protocol).strip().lower()
    if is_lodo_protocol(key):
        cohort_metrics = [
            _regression_oof_metrics(group)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        return {
            "pooled_sample_weighted": _regression_oof_metrics(frame),
            "cohort_macro_equal_weight": _regression_mean_metrics(cohort_metrics),
        }
    repeat_metrics = [
        _regression_oof_metrics(group)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    return {"mean_repeat_pooled_oof": _regression_mean_metrics(repeat_metrics)}


def _resample_regression_subjects(
    frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    subjects = [
        group.reset_index(drop=True)
        for _, group in frame.groupby("_subject_id", sort=True)
    ]
    if not subjects:
        return frame.iloc[0:0].copy()
    chosen = rng.integers(0, len(subjects), size=len(subjects))
    return pd.concat([subjects[int(index)] for index in chosen], ignore_index=True)


def _regression_bootstrap_estimands(
    frame: pd.DataFrame, protocol: str, n_bootstrap: int, rng: np.random.Generator
) -> dict[str, dict[str, np.ndarray]]:
    key = str(protocol).strip().lower()
    names = (
        ("pooled_sample_weighted", "cohort_macro_equal_weight")
        if is_lodo_protocol(key)
        else ("mean_repeat_pooled_oof",)
    )
    storage = {
        name: {
            metric: np.full(int(n_bootstrap), np.nan, dtype=float)
            for metric in REGRESSION_METRIC_COLUMNS
        }
        for name in names
    }
    if frame.empty:
        return storage
    if is_lodo_protocol(key):
        cohorts = [
            group.reset_index(drop=True)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        for i in range(int(n_bootstrap)):
            chosen = rng.integers(0, len(cohorts), size=len(cohorts))
            sampled = [
                _resample_regression_subjects(cohorts[int(index)], rng)
                for index in chosen
            ]
            pooled = _regression_oof_metrics(pd.concat(sampled, ignore_index=True))
            macro = _regression_mean_metrics(
                [_regression_oof_metrics(group) for group in sampled]
            )
            for metric in REGRESSION_METRIC_COLUMNS:
                storage["pooled_sample_weighted"][metric][i] = pooled.get(
                    metric, np.nan
                )
                storage["cohort_macro_equal_weight"][metric][i] = macro.get(
                    metric, np.nan
                )
        return storage
    for i in range(int(n_bootstrap)):
        sampled = _resample_regression_subjects(frame, rng)
        metrics = [
            _regression_oof_metrics(group)
            for _, group in sampled.groupby("_repeat", sort=True)
        ]
        averaged = _regression_mean_metrics(metrics)
        for metric in REGRESSION_METRIC_COLUMNS:
            storage["mean_repeat_pooled_oof"][metric][i] = averaged.get(metric, np.nan)
    return storage


def _paired_regression_prediction_frame(
    a: pd.DataFrame, b: pd.DataFrame, protocol: str
) -> pd.DataFrame:
    left = _prepare_regression_oof(a, protocol)
    right = _prepare_regression_oof(b, protocol)
    if left.empty or right.empty:
        return pd.DataFrame()
    keep_left = [
        "outer_split_key",
        "sample_id",
        "_subject_id",
        "_repeat",
        "_cluster",
        "y_true",
        "y_pred",
    ]
    keep_right = ["outer_split_key", "sample_id", "y_true", "y_pred"]
    merged = left[keep_left].merge(
        right[keep_right],
        on=["outer_split_key", "sample_id"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return merged
    if not np.allclose(
        merged["y_true_a"].to_numpy(dtype=float),
        merged["y_true_b"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Paired regression strategies disagree on held-out targets.")
    return merged


def _paired_regression_oof_ci(
    merged: pd.DataFrame, metric: str, protocol: str, seed: int, n_bootstrap: int = 2000
) -> tuple[float, float, float]:
    if merged.empty:
        return float("nan"), float("nan"), float("nan")

    def metric_value(frame: pd.DataFrame, pred_col: str) -> float:
        values = compute_regression_metrics(
            frame["y_true_a"].to_numpy(dtype=float),
            frame[pred_col].to_numpy(dtype=float),
        )
        return float(values.get(metric, np.nan))

    key = str(protocol).strip().lower()
    if is_lodo_protocol(key):
        estimate = metric_value(merged, "y_pred_a") - metric_value(merged, "y_pred_b")
    else:
        differences = []
        for _, group in merged.groupby("_repeat", sort=True):
            differences.append(
                metric_value(group, "y_pred_a") - metric_value(group, "y_pred_b")
            )
        estimate = float(np.mean(differences)) if differences else float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.full(int(n_bootstrap), np.nan, dtype=float)
    if is_lodo_protocol(key):
        cohorts = [
            group.reset_index(drop=True)
            for _, group in merged.groupby("_cluster", sort=True)
        ]
        for i in range(int(n_bootstrap)):
            chosen = rng.integers(0, len(cohorts), size=len(cohorts))
            sampled = []
            for index in chosen:
                cohort = cohorts[int(index)]
                subjects = [
                    g.reset_index(drop=True)
                    for _, g in cohort.groupby("_subject_id", sort=True)
                ]
                subject_draws = rng.integers(0, len(subjects), size=len(subjects))
                sampled.append(
                    pd.concat(
                        [subjects[int(j)] for j in subject_draws], ignore_index=True
                    )
                )
            boot = pd.concat(sampled, ignore_index=True)
            draws[i] = metric_value(boot, "y_pred_a") - metric_value(boot, "y_pred_b")
    else:
        subjects = [
            group.reset_index(drop=True)
            for _, group in merged.groupby("_subject_id", sort=True)
        ]
        for i in range(int(n_bootstrap)):
            chosen = rng.integers(0, len(subjects), size=len(subjects))
            boot = pd.concat([subjects[int(j)] for j in chosen], ignore_index=True)
            diffs = []
            for _, group in boot.groupby("_repeat", sort=True):
                diffs.append(
                    metric_value(group, "y_pred_a") - metric_value(group, "y_pred_b")
                )
            if diffs:
                draws[i] = float(np.mean(diffs))
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _regression_statistics(
    root: Path, seed: int, protocol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:

    frames = _regression_strategy_frames(root)
    prediction_frames = _regression_prediction_frames(root)

    rows: list[dict[str, Any]] = []
    for strategy, raw in prediction_frames.items():
        frame = _prepare_regression_oof(raw, protocol)
        if frame.empty:
            continue
        point = _regression_point_estimands(frame, protocol)
        boot = _regression_bootstrap_estimands(
            frame,
            protocol,
            2000,
            np.random.default_rng(seed + len(rows) * 101),
        )
        for estimand, metrics in point.items():
            for metric in REGRESSION_METRIC_COLUMNS:
                estimate = float(metrics.get(metric, np.nan))
                if not np.isfinite(estimate):
                    continue
                samples = np.asarray(boot[estimand][metric], dtype=float)
                samples = samples[np.isfinite(samples)]
                low = high = float("nan")
                if len(samples):
                    low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "Strategy": strategy,
                        "estimand": estimand,
                        "metric": metric,
                        "estimate": estimate,
                        "std": float(np.std(samples, ddof=1))
                        if len(samples) > 1
                        else 0.0,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "n_outer_units": int(frame["outer_split_key"].nunique()),
                        "n_subjects": int(frame["_subject_id"].nunique()),
                        "n_bootstrap": 2000,
                    }
                )
    summary = pd.DataFrame(rows)

    pair_rows: list[dict[str, Any]] = []

    strategies = list(frames)

    for i, strategy_a in enumerate(strategies):
        for strategy_b in strategies[i + 1 :]:
            a = frames[strategy_a].copy()

            b = frames[strategy_b].copy()

            key = (
                "outer_split_key"
                if "outer_split_key" in a.columns and "outer_split_key" in b.columns
                else None
            )

            for metric in REGRESSION_METRIC_COLUMNS:
                if metric not in a.columns or metric not in b.columns:
                    continue

                if key:
                    aa = a[[key, metric]].rename(columns={metric: "a"})

                    bb = b[[key, metric]].rename(columns={metric: "b"})

                    matched = aa.merge(bb, on=key, how="inner")

                else:
                    va = _finite(a[metric])

                    vb = _finite(b[metric])

                    n = min(len(va), len(vb))

                    matched = pd.DataFrame(
                        {
                            "outer_split_key": [f"r0_o{i}" for i in range(n)],
                            "a": va[:n],
                            "b": vb[:n],
                        }
                    )

                va = pd.to_numeric(matched["a"], errors="coerce").to_numpy(dtype=float)

                vb = pd.to_numeric(matched["b"], errors="coerce").to_numpy(dtype=float)

                mask = np.isfinite(va) & np.isfinite(vb)

                matched = matched.loc[mask].copy()

                if matched.empty:
                    continue

                diff = pd.to_numeric(matched["a"], errors="coerce").to_numpy(
                    dtype=float
                ) - pd.to_numeric(matched["b"], errors="coerce").to_numpy(dtype=float)

                prediction_pair = pd.DataFrame()
                if strategy_a in prediction_frames and strategy_b in prediction_frames:
                    prediction_pair = _paired_regression_prediction_frame(
                        prediction_frames[strategy_a],
                        prediction_frames[strategy_b],
                        protocol,
                    )
                oof_difference, low, high = _paired_regression_oof_ci(
                    prediction_pair,
                    metric,
                    protocol,
                    seed + 5000 + len(pair_rows) * 101,
                    2000,
                )
                if not np.isfinite(oof_difference):
                    oof_difference = float(np.mean(diff))
                    low, high = _paired_bootstrap_difference(
                        matched,
                        "a",
                        "b",
                        protocol,
                        2000,
                        np.random.default_rng(seed + 5000 + len(pair_rows) * 101),
                    )

                if is_lodo_protocol(protocol):
                    _, p_value, test = _exact_sign_flip_test(
                        diff, seed + 9000 + len(pair_rows) * 101
                    )

                else:
                    _, p_value, test = _corrected_resampled_t_test(matched, "a", "b")

                pair_rows.append(
                    {
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "difference_a_minus_b": float(oof_difference),
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "effect_estimand": "pooled_sample_weighted_oof"
                        if is_lodo_protocol(protocol)
                        else "mean_repeat_pooled_oof",
                        "test_estimand": "paired_outer_cohort_metrics"
                        if is_lodo_protocol(protocol)
                        else "paired_outer_fold_metrics_corrected_resampled_t",
                        "n_matched_outer_units": len(diff),
                        "test": test,
                        "p_value": p_value,
                    }
                )

    pairwise = pd.DataFrame(pair_rows)

    if not pairwise.empty:
        pairwise["p_holm"] = np.nan

        for metric, idx in pairwise.groupby("metric").groups.items():
            indices = list(idx)

            vals = [float(pairwise.at[i, "p_value"]) for i in indices]

            finite_positions = [j for j, value in enumerate(vals) if np.isfinite(value)]

            adjusted = _holm_adjust([vals[j] for j in finite_positions])

            for pos, value in zip(finite_positions, adjusted):
                pairwise.at[indices[pos], "p_holm"] = value

        pairwise["significant_holm_0_05"] = pd.to_numeric(
            pairwise["p_holm"], errors="coerce"
        ).lt(0.05)

    return summary, pairwise


def _format_metric(
    metric: str, estimate: Any, std: Any, low: Any, high: Any, bold: bool = False
) -> str:

    values = []

    for value in (estimate, std, low, high):
        try:
            value = float(value)

        except Exception:
            value = float("nan")

        values.append(value)

    estimate_f, std_f, low_f, high_f = values

    if not np.isfinite(estimate_f):
        return "NA"

    body = f"{estimate_f:.3f}"

    if np.isfinite(std_f):
        body += f" ± {std_f:.3f}"

    if np.isfinite(low_f) and np.isfinite(high_f):
        body += f" [{low_f:.3f}, {high_f:.3f}]"

    return f"<strong>{body}</strong>" if bold else body


def _regression_estimand_label(value: Any) -> str:

    labels = {
        "mean_repeat_pooled_oof": "Mean repeat-level pooled OOF",
        "pooled_sample_weighted": "Pooled sample-weighted OOF",
        "cohort_macro_equal_weight": "Equal-cohort macro OOF",
    }

    text = str(value)

    return labels.get(text, text)


def _regression_performance_table(
    statistics: pd.DataFrame, primary_metric: str = "RMSE"
) -> pd.DataFrame:

    if statistics.empty:
        return pd.DataFrame()

    working = statistics.copy()

    if "estimand" not in working.columns:
        working["estimand"] = "legacy_outer_unit_mean"

    primary = str(primary_metric).strip()

    display_metrics = [primary] + [
        metric for metric in _REGRESSION_DISPLAY_METRICS if metric != primary
    ]

    units = (
        working[["Strategy", "estimand"]]
        .dropna(subset=["Strategy", "estimand"])
        .astype(str)
        .drop_duplicates()
    )

    best: dict[tuple[str, str], str] = {}

    for estimand in units["estimand"].drop_duplicates().tolist():
        for metric in display_metrics:
            sub = working[
                working["estimand"].astype(str).eq(str(estimand))
                & working["metric"].astype(str).eq(metric)
            ].copy()

            sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")

            sub = sub[np.isfinite(sub["estimate"].to_numpy(dtype=float))]

            if sub.empty:
                continue

            sub = sub.sort_values("estimate", ascending=metric_is_loss(metric))

            best[(str(estimand), metric)] = str(sub.iloc[0]["Strategy"])

    rows = []

    show_estimand = (
        units["estimand"].nunique() > 1
        or not units["estimand"].eq("mean_repeat_pooled_oof").all()
    )

    for item in units.to_dict(orient="records"):
        strategy = str(item["Strategy"])
        estimand = str(item["estimand"])
        row: dict[str, Any] = {"Strategy": strategy}

        if show_estimand:
            row["Estimand"] = _regression_estimand_label(estimand)

        for metric in display_metrics:
            sub = working[
                working["Strategy"].astype(str).eq(strategy)
                & working["estimand"].astype(str).eq(estimand)
                & working["metric"].astype(str).eq(metric)
            ]

            if sub.empty:
                row[metric] = "NA"

                continue

            item_metric = sub.iloc[0]

            row[metric] = _format_metric(
                metric,
                item_metric.get("estimate"),
                item_metric.get("std"),
                item_metric.get("ci_low"),
                item_metric.get("ci_high"),
                bold=best.get((estimand, metric)) == strategy,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _regression_mean_std_cell(mean: Any, std: Any) -> str:

    try:
        mean_f = float(mean)

    except Exception:
        mean_f = float("nan")

    try:
        std_f = float(std)

    except Exception:
        std_f = float("nan")

    if not np.isfinite(mean_f):
        return "NA"

    if np.isfinite(std_f):
        return f"{mean_f:.3f} ± {std_f:.3f}"

    return f"{mean_f:.3f}"


def _regression_top_mpmas(
    root: Path,
    metric: str,
    n: int = 5,
    learner_labels: dict[str, str] | None = None,
) -> pd.DataFrame:

    inner = _read_table(root / "inner_results" / "inner_results.parquet")

    outer = _read_table(root / "results" / "outer_results.parquet")

    if inner.empty:
        return pd.DataFrame()

    if "ok" in inner.columns:
        inner = inner[
            pd.to_numeric(inner["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        ]

    if "ok" in outer.columns and not outer.empty:
        outer = outer[
            pd.to_numeric(outer["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        ]

    ids = [
        c
        for c in (
            "config_id",
            "candidate_family",
            "modalities",
            "integration",
            "integration_n_components",
            "count_transformation",
            "resolution",
            "learner",
        )
        if c in inner.columns
    ]

    available = [c for c in _REGRESSION_DISPLAY_METRICS if c in inner.columns]

    if metric not in available:
        metric = (
            "RMSE" if "RMSE" in available else (available[0] if available else metric)
        )

    metric_order = [metric] + [c for c in available if c != metric]

    inner_agg = (
        inner.groupby(ids, dropna=False)[metric_order]
        .agg(["mean", "std"])
        .reset_index()
    )

    inner_agg.columns = [
        c if isinstance(c, str) else c[0] if not c[1] else f"inner_{c[0]}_{c[1]}"
        for c in inner_agg.columns
    ]

    if not outer.empty:
        outer_metrics = [c for c in metric_order if c in outer.columns]

        outer_agg = (
            outer.groupby("config_id", dropna=False)[outer_metrics]
            .agg(["mean", "std"])
            .reset_index()
        )

        outer_agg.columns = [
            c if isinstance(c, str) else c[0] if not c[1] else f"outer_{c[0]}_{c[1]}"
            for c in outer_agg.columns
        ]

        inner_agg = inner_agg.merge(outer_agg, on="config_id", how="left")

    sort_col = f"inner_{metric}_mean"

    if sort_col in inner_agg.columns:
        inner_agg = inner_agg.sort_values(sort_col, ascending=metric_is_loss(metric))

    rows = []

    labels = {"R2": "R²", "PearsonR": "Pearson r", "SpearmanR": "Spearman ρ"}

    for rank, (_, item) in enumerate(inner_agg.head(int(n)).iterrows(), start=1):
        row: dict[str, Any] = {"Rank": rank}

        if "candidate_family" in inner_agg.columns:
            row["Family"] = item.get("candidate_family", "")

            row["Modalities"] = item.get("modalities", "")

            row["Integration"] = item.get("integration", "")

            components = item.get("integration_n_components", "")

            row["Components"] = "" if pd.isna(components) else components

            row["Representation"] = _report._display_token(item.get("resolution", ""))

            row["Transformation"] = _report._display_token(
                item.get("count_transformation", "")
            )

        else:
            row["Taxonomic representation"] = _report._display_token(
                item.get("resolution", "")
            )

            row["Count transformation"] = _report._display_token(
                item.get("count_transformation", "")
            )

        row["Learner"] = _report._display_learner(
            item.get("learner", ""), learner_labels
        )

        for name in metric_order:
            label = labels.get(name, name)

            row[f"Inner {label}"] = _regression_mean_std_cell(
                item.get(f"inner_{name}_mean"), item.get(f"inner_{name}_std")
            )

            row[f"Outer {label}"] = _regression_mean_std_cell(
                item.get(f"outer_{name}_mean"), item.get(f"outer_{name}_std")
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _task_data_format(sweep: Sweep) -> str:

    if getattr(sweep, "uses_modalities", False):
        return (
            "multimodal"
            if len(getattr(sweep, "modalities", ())) > 1
            else "single-modality"
        )

    source = getattr(sweep, "data", None)

    return _report._display_data_format(getattr(source, "format", ""))


def _regression_performance_note(protocol: Any, n_bootstrap: int = 2000) -> str:

    key = str(protocol).strip().lower()

    if is_lodo_protocol(key):
        uncertainty = f"Regression metrics are calculated directly from held-out out-of-fold predictions. Both pooled sample-weighted and equal-cohort macro estimands are reported. Their 95% percentile confidence intervals and displayed bootstrap standard deviations use {int(n_bootstrap):,} two-stage replicates that resample held-out cohorts and then subjects within each sampled cohort."

    else:
        uncertainty = f"Regression metrics are calculated directly from held-out out-of-fold predictions within each repeat and averaged equally across repeats. Their 95% percentile confidence intervals and displayed bootstrap standard deviations use {int(n_bootstrap):,} subject-cluster replicates; every sampled subject carries all of its repeat-specific held-out predictions together."

    metrics = " RMSE is the square root of mean squared error. MAE is mean absolute error. R² compares explained variation with a mean-prediction baseline. Pearson r measures linear association. Spearman ρ measures rank association. Lower RMSE and MAE are better. Higher R² and correlations are better."

    return f"<p>{html.escape(uncertainty + metrics)}</p>"


def _regression_procedure(sweep: Sweep, root: Path) -> pd.DataFrame:

    manifest = _read_json(root / "manifest.json")

    ev = sweep.evaluation

    gate = sweep.gate

    target = (
        sweep.samples.target_col
        if getattr(sweep, "uses_modalities", False)
        else sweep.data.target_col
    )

    rows = [
        ["Task", sweep.title],
        ["Task type", "Regression"],
        ["Target", str(target)],
        ["Experiment directory", str(root)],
        ["Data format", _task_data_format(sweep)],
        ["Samples", str(manifest.get("n_samples", ""))],
    ]

    fold_pca_target = getattr(sweep.data, "fold_pca_target", None)
    if fold_pca_target is not None:
        rows.append(
            ["Target construction", _fold_pca_target_description(fold_pca_target)]
        )

    if getattr(sweep, "uses_modalities", False):
        rows.extend(
            [
                ["Primary modality", str(manifest.get("primary_modality", ""))],
                [
                    "Modalities",
                    ", ".join(str(x) for x in manifest.get("modalities", {}).keys()),
                ],
            ]
        )

    rows.extend(
        [
            ["Procedure", ev.protocol],
            ["Benchmark", ev.benchmark_id or ""],
            [
                "Outer folds",
                ev.outer_folds
                if not is_lodo_protocol(ev.protocol)
                else f"LODO ({ev.outer_folds} datasets)",
            ],
            [
                "Inner folds",
                ev.inner_folds
                if not is_lodo_protocol(ev.protocol)
                else f"{ev.inner_folds}-fold {str(ev.inner_grouping).replace('_', ' ')} inner CV",
            ],
            ["Repeats", ev.repeats],
            ["Selection metric", ev.optimize_metric],
            ["Random seed", ev.random_state],
            [
                "Qualification gate",
                f"on ({gate.metric} threshold {gate.threshold})"
                if gate.enabled
                else "off",
            ],
            [
                "Selection rule",
                "Reported performance uses held-out outer evaluation predictions.",
            ],
        ]
    )

    return pd.DataFrame(rows, columns=["Field", "Value"])


def _fold_pca_target_description(spec: Any) -> str:
    mapping = "; ".join(
        f"{str(domain).replace('_', ' ')} = {component}"
        for domain, component in spec.domain_to_component
    )
    n_questionnaires = len(spec.questionnaire_columns)
    reference_size = (
        f"the independent {int(spec.reference_n)}-participant reference cohort"
        if spec.reference_n is not None
        else "the independent reference cohort"
    )
    return (
        f"For each inner and outer split, fit {int(spec.n_components)}-component PCA "
        f"to the {n_questionnaires} questionnaire subscales using that split's training "
        "participants only. Standardize with training-fold means and SDs, apply "
        f"Promax rotation (power {int(spec.promax_power)}), and calculate regression "
        "factor scores for both training and held-out participants with the fitted "
        f"fold transform. Match and orient fold components to {reference_size} "
        "loadings by maximum absolute Tucker congruence. Reference mapping: "
        f"{mapping}. The four domain scores are fitted as separate regression targets."
    )


def _pca_alignment_stability_table(children: list[Sweep]) -> pd.DataFrame:
    """Summarize selected MPMA-B fold PCA alignment diagnostics by domain."""
    rows: list[dict[str, Any]] = []
    for child in children:
        spec = getattr(child.data, "fold_pca_target", None)
        if spec is None:
            continue
        target = str(child.data.target_col)
        component = spec.reference_components[spec.domain_index(target)]
        root = Path(child.root())
        raw = _read_table(root / "results" / "outer_results.parquet")
        selected = _read_table(root / "tables" / "mpma_b_outer_selection.parquet")
        congruence_column = f"pca_congruence_{component}"
        if raw.empty or selected.empty or congruence_column not in raw.columns:
            continue
        if not {"split_key", "config_id"}.issubset(raw.columns):
            continue
        if not {"outer_split_key", "config_id"}.issubset(selected.columns):
            continue

        chosen = selected[["outer_split_key", "config_id"]].copy()
        chosen["outer_split_key"] = chosen["outer_split_key"].astype(str)
        chosen["config_id"] = chosen["config_id"].astype(str)
        raw = raw.copy()
        raw["split_key"] = raw["split_key"].astype(str)
        raw["config_id"] = raw["config_id"].astype(str)
        fold_rows = raw.merge(
            chosen,
            left_on=["split_key", "config_id"],
            right_on=["outer_split_key", "config_id"],
            how="inner",
            validate="one_to_one",
        )
        if fold_rows.empty:
            continue

        values = pd.to_numeric(fold_rows[congruence_column], errors="coerce")
        valid = values.notna() & np.isfinite(values.to_numpy(dtype=float))
        fold_rows = fold_rows.loc[valid].copy()
        values = values.loc[valid].to_numpy(dtype=float)
        if len(values) == 0:
            continue

        match_column = f"pca_matched_fold_component_{component}"
        match_values = (
            pd.to_numeric(fold_rows[match_column], errors="coerce").dropna()
            if match_column in fold_rows.columns
            else pd.Series(dtype=float)
        )
        sign_column = f"pca_sign_applied_{component}"
        sign_flips = (
            int(pd.to_numeric(fold_rows[sign_column], errors="coerce").fillna(0).sum())
            if sign_column in fold_rows.columns
            else 0
        )
        match_mode = (
            int(match_values.mode().iloc[0]) if not match_values.empty else np.nan
        )
        match_consistency = (
            float((match_values == match_mode).mean() * 100.0)
            if not match_values.empty and pd.notna(match_mode)
            else np.nan
        )
        rows.append(
            {
                "Domain": target.replace("_", " "),
                "Reference component": component,
                "Outer folds": int(len(values)),
                "Mean absolute congruence": float(np.mean(values)),
                "SD absolute congruence": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "Minimum absolute congruence": float(np.min(values)),
                "Folds at or above threshold (%)": float(
                    np.mean(values >= float(spec.min_alignment_congruence)) * 100.0
                ),
                "Threshold": float(spec.min_alignment_congruence),
                "Most common matched fold component": match_mode,
                "Match consistency (%)": match_consistency,
                "Fold sign adjustments": sign_flips,
            }
        )
    return pd.DataFrame(rows)


def _pca_example_subject_predictions(
    children: list[Sweep], subject_id: str = "SYN0001"
) -> pd.DataFrame:
    """Collect MPMA-B held-out scores for a subject, averaged across CV repeats."""
    records: list[dict[str, Any]] = []
    for child in children:
        target = str(child.data.target_col)
        root = Path(child.root())
        frame = _read_table(root / "predictions" / "mpma_b_outer_predictions.parquet")
        required = {"sample_id", "outer_split_key", "y_true", "y_pred"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        sample_match = frame["sample_id"].astype(str) == str(subject_id)
        if not sample_match.any() and "subject_id" in frame.columns:
            sample_match = frame["subject_id"].astype(str) == str(subject_id)
        frame = frame.loc[sample_match].copy()
        if frame.empty:
            continue
        frame["repeat"] = frame["outer_split_key"].map(_regression_repeat_id)
        frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
        frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
        frame = frame.dropna(subset=["y_true", "y_pred"])
        if frame.empty:
            continue
        per_repeat = frame.groupby("repeat", as_index=False).agg(
            true_score=("y_true", "mean"), predicted_score=("y_pred", "mean")
        )
        for row in per_repeat.to_dict(orient="records"):
            records.append(
                {
                    "Subject": str(subject_id),
                    "Domain": target,
                    "Repeat": str(row["repeat"]),
                    "True score": float(row["true_score"]),
                    "Predicted score": float(row["predicted_score"]),
                }
            )
    per_repeat = pd.DataFrame(records)
    if per_repeat.empty:
        return per_repeat
    summary = (
        per_repeat.groupby(["Subject", "Domain"], as_index=False)
        .agg(
            True_score=("True score", "mean"),
            Predicted_score=("Predicted score", "mean"),
            Repeats=("Repeat", "nunique"),
        )
        .rename(
            columns={
                "True_score": "Mean true score",
                "Predicted_score": "Mean predicted score",
            }
        )
    )
    repeat_counts = per_repeat.groupby("Domain")["Repeat"].nunique()
    summary["Repeat count"] = summary["Domain"].map(repeat_counts).astype(int)
    return summary


def _write_pca_example_radar(
    summary: pd.DataFrame, report_dir: Path, domain_order: list[str]
) -> Path | None:
    if summary.empty or not set(domain_order).issubset(set(summary["Domain"].astype(str))):
        return None
    ordered = summary.set_index("Domain").loc[domain_order]
    labels = [target.replace("_", " ") for target in domain_order]
    truth = ordered["Mean true score"].to_numpy(dtype=float)
    prediction = ordered["Mean predicted score"].to_numpy(dtype=float)
    lower = min(0.0, float(np.min(np.r_[truth, prediction])))
    upper = max(0.0, float(np.max(np.r_[truth, prediction])))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        return None
    offset = lower
    radial_truth = truth - offset
    radial_prediction = prediction - offset
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    truth_closed = np.r_[radial_truth, radial_truth[0]]
    prediction_closed = np.r_[radial_prediction, radial_prediction[0]]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "pca_example_subject_radar.png"
    fig, ax = plt.subplots(figsize=(7.5, 6.5), subplot_kw={"projection": "polar"})
    ax.plot(closed_angles, truth_closed, marker="o", linewidth=2, label="True score")
    ax.fill(closed_angles, truth_closed, alpha=0.12)
    ax.plot(
        closed_angles,
        prediction_closed,
        marker="o",
        linewidth=2,
        label="MPMA-B predicted score",
    )
    ax.fill(closed_angles, prediction_closed, alpha=0.10)
    ax.set_xticks(angles, labels)
    tick_values = np.linspace(lower, upper, 5)
    tick_values = np.unique(np.round(tick_values, 3))
    tick_values = tick_values[tick_values >= lower]
    ax.set_yticks(tick_values - offset, [f"{value:.2f}" for value in tick_values])
    ax.set_ylim(0.0, upper - offset)
    ax.set_rlabel_position(22.5)
    ax.set_title("SYN0001: held-out domain scores, averaged across repeats", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12))
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _regression_outer_unit_table(root: Path) -> pd.DataFrame:

    frames = []

    for strategy, frame in _regression_strategy_frames(root).items():
        if frame.empty:
            continue

        out = frame.copy()

        if "outer_split_key" not in out.columns and "split_key" in out.columns:
            out["outer_split_key"] = out["split_key"].astype(str)

        out.insert(0, "Strategy", strategy)

        keep = [
            c
            for c in ("Strategy", "outer_split_key", *REGRESSION_METRIC_COLUMNS)
            if c in out.columns
        ]

        frames.append(out[keep])

    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def _regression_fixed(value: Any, digits: int = 3) -> str:

    try:
        number = float(value)

    except Exception:
        return ""

    if not np.isfinite(number):
        return ""

    return f"{number:.{int(digits)}f}"


def _regression_numeric_table(table: pd.DataFrame, limit: int) -> pd.DataFrame:

    if table.empty:
        return pd.DataFrame()

    out = table.head(int(limit)).copy()

    rename = {}

    for column in out.columns:
        text = str(column).replace("_", " ").strip()

        rename[column] = text[:1].upper() + text[1:] if text else str(column)

    out = out.rename(columns=rename)

    for column in out.columns:
        if "feature" in column.casefold() or "taxon" in column.casefold():
            out[column] = out[column].map(_report._short_feature_label)

            continue

        original = next((src for src, dst in rename.items() if dst == column), column)

        numeric = (
            pd.to_numeric(table[original].head(int(limit)), errors="coerce")
            if original in table.columns
            else pd.Series(dtype=float)
        )

        if len(numeric) == len(out) and numeric.notna().any():
            if numeric.dropna().map(lambda x: float(x).is_integer()).all():
                out[column] = numeric.map(lambda x: "" if pd.isna(x) else str(int(x)))

            else:
                out[column] = numeric.map(lambda x: _regression_fixed(x, 3))

    return out


def _regression_explainability_blocks(
    root: Path, report_dir: Path, top_k: int
) -> tuple[str, int]:

    parts: list[str] = []

    count = 0

    for slug, label in (("mpma_b", "MPMA-B"), ("mpma_e", "MPMA-E")):
        target_dir = root / "explainability" / slug

        stability = _read_table(target_dir / "feature_stability.parquet")

        combined = _read_table(target_dir / "feature_importance.parquet")

        meta = _read_json(target_dir / "explained_unit.json")

        if not target_dir.exists() or (stability.empty and combined.empty):
            continue

        count += 1

        methods = [str(x) for x in meta.get("methods", []) if str(x)]

        parts.append(f'<section class="xai-target"><h3>{html.escape(label)}</h3>')

        if methods:
            parts.append(
                f"<p>Cross-fitted out-of-fold regression explanations of the final selected specification. Methods: {html.escape(', '.join(methods))}.</p>"
            )

        _write_regression_explainability_figures(target_dir, stability, int(top_k))

        figure = _report._fig(
            target_dir / "figures" / "feature_support",
            report_dir,
            f"{label}: cross-method top-k support and fold stability",
        )

        if figure:
            parts.append("<h4>Global explanations</h4>")

            parts.append(
                "<p>Top-k support is a within-method rank score: rank 1 scores 1, rank k scores 1/k, and ranks below k score 0. Fold stability is the fraction of estimable outer folds in which a feature ranks within the method-specific top k. Both are scale-free 0–1 summaries. Method-specific effect magnitudes are not compared across methods.</p>"
            )

            parts.append(figure)

        curves = _read_table(target_dir / "ale_curves.parquet")

        if not curves.empty:
            parts.append("<h4>ALE</h4>")

            parts.append(
                "<p>Accumulated local effects summarize how the fitted regression prediction changes over the observed feature range. Thin pale lines show the outer-fold ALE curves. When multiple displayed outer-fold curves have a shared feature-value range, the thick blue line shows their pointwise median only over that common-support range and the shaded band shows the corresponding interquartile range across folds. Individual fold curves may extend beyond the common-support range. If no shared range exists, no cross-fold median or interquartile band is drawn. If only one ALE curve is available, it is shown without an interquartile band. The dashed horizontal line marks zero centered ALE effect, so values above or below it indicate predictions higher or lower than the centered reference in target units. ALE describes model behavior and should not be interpreted as a causal effect.</p>"
            )

            curve_figure = _report._fig(
                target_dir / "figures" / "ale_curves",
                report_dir,
                f"{label}: ALE effect curves",
            )

            if curve_figure:
                parts.append(curve_figure)

            parts.append(
                "<p>Per-fold numerical ALE coordinates are retained in the technical artifact <code>ale_curves.parquet</code>.</p>"
            )

        interactions = _read_table(target_dir / "ale_interactions.parquet")

        if not interactions.empty:
            parts.append("<h4>ALE interactions</h4>")

            parts.append(
                "<p>Exploratory out-of-fold 2D ALE interaction strengths. Edge width and shade represent interaction magnitude. Node area encodes cohort mean relative abundance on a log scale; for log-ratio and balance coordinates, abundance is the absolute-coefficient-weighted mean relative abundance of the underlying component taxa.</p>"
            )

            for stem, caption in (
                ("interaction_network_current", "2D ALE interaction network"),
            ):
                block = _report._fig(
                    target_dir / "figures" / stem, report_dir, f"{label}: {caption}"
                )

                if block:
                    parts.append(block)

            parts.append("<h5>Interaction strengths</h5>")

            parts.append(
                _report._html_table(_regression_numeric_table(interactions, int(top_k)))
            )

        local_mode = str(meta.get("local_explanations", "none")).strip() or "none"

        parts.append("<h4>Local explanations</h4>")

        if local_mode == "none":
            parts.append(
                "<p>Local sample-level reporting was not requested for this run.</p>"
            )

        else:
            local_figure = _report._xai_local_figure(
                target_dir, report_dir, label, "regression"
            )

            if local_figure:
                parts.append(
                    "<p>Representative held-out samples are selected from the out-of-fold prediction distribution at the lower, central, and upper response ranges. The cohort relative abundance panel shows the raw cohort relative abundance of each displayed taxon and outlines the explained sample. Each sample is explained only by outer-fold models that did not train on that sample. SHAP attributions and LIME local-surrogate coefficients are shown side by side when both are available. The cross-method panel combines attribution direction with within-method reciprocal-rank support and does not average raw SHAP and LIME magnitudes. The support count indicates how many local methods place the feature within the displayed top set.</p>"
                )

                parts.append(local_figure)

                parts.append(
                    "<p>Sample-wise local attribution results are retained in <code>local_explanations.parquet</code>.</p>"
                )

            else:
                parts.append(
                    "<p>No representative local SHAP/LIME visualization is available for this explained unit.</p>"
                )

        parts.append("</section>")

    return "".join(parts), count


def write_regression_report(sweep: Sweep, emit_html: bool = True) -> dict[str, Path]:

    root = Path(sweep.root())

    source = root / "results" / "outer_results.parquet"

    if not table_exists(source):
        raise FileNotFoundError("Run evaluate(sweep) before reporting.")

    if (
        not (root / "final_models.json").exists()
        and (root / "tables" / "mpma_b_final_candidate.json").exists()
    ):
        build_final_models(root)

    report_dir = root / "report"

    report_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = report_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    stage("Report", str(report_dir))

    with phase_progress("Regression report analysis", 5) as phase:
        phase.phase("procedure and statistics")

        metric = str(sweep.evaluation.optimize_metric)

        procedure = _regression_procedure(sweep, root)

        statistics, pairwise = _regression_statistics(
            root, int(sweep.evaluation.random_state), str(sweep.evaluation.protocol)
        )

        performance = _regression_performance_table(statistics, metric)

        phase.phase("rankings and ensemble summaries")

        learner_labels = _report._learner_display_map(sweep)

        top_mpmas = _regression_top_mpmas(root, metric, 5, learner_labels)

        ensemble_summary = _report._ensemble_summary_table(root)

        ensemble_members = _report._ensemble_members_table(root, learner_labels)

        strategy_rows = [{"Strategy": x} for x in _regression_strategy_frames(root)]

        phase.phase("compute accounting")

        compute = run_compute_accounting(root, sweep, strategy_rows)

        compute_display = compute.get("display", pd.DataFrame())

        compute_environment = compute.get("environment", {})

        phase.phase("outer-unit summaries")

        outer_units = _regression_outer_unit_table(root)

        phase.phase("explainability visuals")

        explainability_html, explainability_count = _regression_explainability_blocks(
            root, report_dir, int(getattr(sweep.explainability, "top_k", 15))
        )

    feature_summary = _report._feature_support_table(root, top_n=8)

    feature_summary_path = tables_dir / "important_features.parquet"

    write_table(feature_summary_path, feature_summary)

    hardware_summary = _report._hardware_summary_table(compute_environment)

    hardware_summary_path = tables_dir / "hardware_environment.parquet"

    write_table(hardware_summary_path, hardware_summary)

    primary_pairwise = _report._primary_pairwise_display(pairwise, metric)

    statistics_path = tables_dir / "strategy_metrics_bootstrap.parquet"

    pairwise_path = tables_dir / "strategy_pairwise_tests.parquet"

    outer_units_path = tables_dir / "strategy_outer_unit_metrics.parquet"

    statistics_manifest_path = tables_dir / "strategy_statistics_manifest.json"

    with phase_progress("Regression report outputs", 2) as phase:
        phase.phase("statistical tables")

        write_table(statistics_path, statistics)

        write_table(pairwise_path, pairwise)

        write_table(outer_units_path, outer_units)

        phase.phase("statistics manifest")

        dump_json_standard(
            {
                "task": "regression",
                "primary_metric": metric,
                "display_metrics": list(_REGRESSION_DISPLAY_METRICS),
                "outer_unit_metrics": outer_units_path,
                "summary": statistics_path,
                "pairwise": pairwise_path,
                "n_bootstrap": 2000,
                "nested_cv_bootstrap": "subject-cluster bootstrap preserving all repeat-specific held-out predictions per sampled subject",
                "lodo_bootstrap": "two-stage held-out cohort and subject-cluster bootstrap",
                "regression_estimands": "metrics computed directly from held-out OOF predictions; repeated CV averages repeat-level pooled OOF metrics and LODO reports pooled sample-weighted plus equal-cohort macro estimands",
                "nested_cv_pairwise_test": "Nadeau–Bengio corrected resampled paired t-test",
                "lodo_pairwise_test": "paired sign-flip randomization test",
                "multiple_testing": "Holm adjustment across strategy pairs within each metric",
            },
            statistics_manifest_path,
        )

    representation_fig = _report._fig(
        root / "figures" / "representation_impact",
        report_dir,
        "Data representations and learner impact on held-out performance",
    )

    mpma_e_fig = _report._fig(
        root / "figures" / "mpma_e", report_dir, "Selected MPMA-E schematic"
    )

    css = _report._report_css()

    raw_metric_cols = {c for c in performance.columns if c != "Strategy"}

    mpma_b = _mpma_b_composition(root, learner_labels)

    ensemble_html = (
        _report._html_table(ensemble_summary) + _report._html_table(ensemble_members)
        if not ensemble_summary.empty or not ensemble_members.empty
        else "<p>Ensemble results are not available yet. Run the ensemble stage and regenerate the report.</p>"
    )

    pairwise_html = (
        _report._html_table(primary_pairwise)
        if not primary_pairwise.empty
        else "<p>Pairwise strategy comparisons require at least two evaluated strategies.</p>"
    )

    robustness_html = _report._robustness_report_html(root)

    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mllabiome report</title><link rel="icon" type="image/svg+xml" href="{_report._favicon_href()}"><style>{css}</style></head>
<body>{_report._report_nav_html(bool(robustness_html))}
<div class="report-shell"><main id="top" class="report-content">
<h2 id="performance-evaluation" class="first-section">Task definition and evaluation procedure</h2>{_report._procedure_grid_html(procedure)}
<h2 id="performance-summary">Task performance summary</h2>{_regression_performance_note(sweep.evaluation.protocol, 2000)}{_report._html_table(performance, raw_html_cols=raw_metric_cols)}
<h2 id="representation-impact">Data representations and learner impact on performance</h2>{_report._representation_impact_note(metric, bool(getattr(sweep, "uses_modalities", False)))}{representation_fig if representation_fig else "<p>No representation-impact figure is available yet.</p>"}
<section id="mpma-b-composition"><h3>Final MPMA-B specification</h3>{_report._html_table(mpma_b)}</section>
<section id="mpma-e-specification"><h3 id="mpma-e">Final MPMA-E specification</h3>{mpma_e_fig}{ensemble_html}</section>
<h2 id="top-mpmas">{_report._top_mpma_heading(top_mpmas)}</h2>{_report._html_table(top_mpmas)}
<h3 id="statistics">Outer-unit strategy comparisons</h3>{_report._inferential_layer_html(sweep.evaluation.protocol, metric)}{pairwise_html}
<h2 id="explainability">Explainability</h2>
<h3>Feature attribution</h3>{explainability_html if explainability_html else "<p>No explainability artefacts are available yet.</p>"}
{robustness_html}
<h2 id="compute">Computational resources</h2><p>Compute is summarized by additive CPU core-hours, model-fit count, and peak resident memory for the worker process tree. CPU time includes child processes and external R processes when used. MPMA-B and MPMA-E share the MPMA search pool, so their compute totals overlap.</p>{_report._html_table(compute_display)}
<h4>Hardware and runtime environment</h4>{_report._html_table(hardware_summary) if not hardware_summary.empty else "<p>Hardware details are unavailable for this run.</p>"}
{_report._abbreviations_html()}
{_report._report_footer_html()}</main></div></body></html>'''

    html_text = _report._sanitize_report_html(html_text)

    html_path = prepare_report_html_path(root, emit_html=emit_html)

    if emit_html:
        html_path.write_text(html_text, encoding="utf-8")

    manifest_path = report_dir / "report_manifest.json"

    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": "regression",
            "report_template": _report.REPORT_TEMPLATE_VERSION,
            "report_layers": list(_report.REPORT_LAYERS),
            "target": str(
                sweep.samples.target_col
                if getattr(sweep, "uses_modalities", False)
                else sweep.data.target_col
            ),
            "figures_embedded": len([f for f in [representation_fig, mpma_e_fig] if f]),
            "explainability_targets": explainability_count,
        },
        manifest_path,
    )

    outputs = {
        "strategy_outer_unit_metrics": outer_units_path,
        "strategy_metrics_bootstrap": statistics_path,
        "strategy_pairwise_tests": pairwise_path,
        "strategy_statistics_manifest": statistics_manifest_path,
        "strategy_compute": compute.get(
            "compute_path", tables_dir / "strategy_compute.parquet"
        ),
        "compute_accounting_manifest": compute.get(
            "manifest_path", tables_dir / "compute_accounting_manifest.json"
        ),
        "important_features": feature_summary_path,
        "hardware_environment": hardware_summary_path,
    }
    if emit_html:
        outputs = {"html_report": html_path, **outputs}

    _report._print_report_summary(
        sweep,
        root,
        procedure,
        performance,
        top_mpmas,
        ensemble_summary,
        ensemble_members,
        primary_pairwise=primary_pairwise,
        compute_display=compute_display,
        compute_environment=compute_environment,
    )

    success("Report completed")

    path_table("Report outputs", outputs)

    return outputs


def _target_primary_summary(child: Sweep) -> dict[str, Any]:

    root = child.root()

    target = str(child.data.target_col)

    task = str(child.data.task)

    metric = str(child.evaluation.optimize_metric)

    rows: dict[str, Any] = {"Target": target, "Task": task, "Primary metric": metric}

    files = {
        "MPMA-B": root / "results" / "mpma_b_outer_results.parquet",
        "MPMA-E": root / "ensembling" / "mpma_e_outer_results.parquet",
    }

    for strategy, path in files.items():
        frame = _read_table(path)

        if frame.empty or metric not in frame.columns:
            rows[strategy] = "NA"

            continue

        values = _finite(frame[metric])

        rows[strategy] = (
            "NA"
            if len(values) == 0
            else f"{float(np.mean(values)):.3f} ± {float(np.std(values, ddof=1)) if len(values) > 1 else 0.0:.3f}"
        )

    return rows


def _section_slug(value: str) -> str:

    return (
        "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-")
        or "target"
    )


def _target_report_section(
    child: Sweep, parent_report_dir: Path, include_nav_landmarks: bool = False
) -> str:

    root = Path(child.root())

    target = str(child.data.target_col)

    task = _target_task(child.data, target)

    section_id = f"target-{_section_slug(target)}"

    tables_dir = root / "report" / "tables"

    metric = str(child.evaluation.optimize_metric)

    learner_labels = _report._learner_display_map(child)

    if task == "regression":
        procedure = _regression_procedure(child, root)

        statistics = _read_table(tables_dir / "strategy_metrics_bootstrap.parquet")

        performance = _regression_performance_table(statistics, metric)

        top_mpmas = _regression_top_mpmas(root, metric, 5, learner_labels)

    else:
        procedure = _report._procedure_table(child, root)

        performance = _report._strategy_performance_display(
            root, selection_metric=metric, html_mode=True
        )

        top_mpmas = _report._top_mpma_display(
            _report._top_mpma_raw(root, 5, metric),
            selection_metric=metric,
            html_mode=True,
            learner_labels=learner_labels,
        )

        if getattr(child, "uses_modalities", False) and not top_mpmas.empty:
            top_mpmas = top_mpmas.rename(
                columns={
                    "Resolution": "Representation",
                    "Count transformation": "Transformation",
                }
            )

    selection = _report._ensemble_summary_table(root)

    members = _report._ensemble_members_table(root, learner_labels)

    pairwise = _report._primary_pairwise_display(
        _read_table(tables_dir / "strategy_pairwise_tests.parquet"), metric
    )

    compute = compute_display(_read_table(tables_dir / "strategy_compute.parquet"))

    feature_summary = _report._feature_support_table(root, top_n=8)

    run_summary = _read_json(root / "run_summary.json")

    machine = (
        run_summary.get("machine", {})
        if isinstance(run_summary.get("machine"), dict)
        else {}
    )

    compute_environment = {
        **machine,
        "evaluation_workers": run_summary.get("workers"),
        "threads_per_worker": run_summary.get("threads_per_worker"),
        "latest_evaluation_invocation_wall_time_s": run_summary.get("elapsed_s"),
    }

    hardware_summary = _report._hardware_summary_table(compute_environment)

    if task == "regression":
        xai, _ = _regression_explainability_blocks(
            root, parent_report_dir, int(getattr(child.explainability, "top_k", 15))
        )

    else:
        xai, _ = _report._explainability_report_blocks(
            root,
            parent_report_dir,
            top_n=int(getattr(child.explainability, "top_k", 15)),
        )

    representation_fig = _report._fig(
        root / "figures" / "representation_impact",
        parent_report_dir,
        f"{target}: data representations and learner impact on held-out performance",
    )

    mpma_e_fig = _report._fig(
        root / "figures" / "mpma_e",
        parent_report_dir,
        f"{target}: selected MPMA-E schematic",
    )

    performance_raw = {c for c in performance.columns if c != "Strategy"}

    top_raw = {
        c for c in top_mpmas.columns if c.startswith("Inner ") or c.startswith("Outer ")
    }

    mpma_b = _mpma_b_composition(root, learner_labels)

    explainability_heading = (
        '<h3 id="explainability">Explainability</h3>'
        if include_nav_landmarks
        else "<h3>Explainability</h3>"
    )

    robustness_html = _report._robustness_report_html(root)

    compute_heading = (
        '<h3 id="compute">Computational resources</h3>'
        if include_nav_landmarks
        else "<h3>Computational resources</h3>"
    )

    oof = ""

    if task != "regression":
        labels = getattr(child.data, "class_labels", None)

        try:
            n_classes = len(labels) if labels is not None else None

        except TypeError:
            n_classes = None

        oof = oof_section_html(
            root / "report",
            n_classes=n_classes,
            heading_level=3,
            id_prefix=section_id,
            include_downloads=False,
        )

    return f'''<section id="{section_id}" class="xai-target"><h2>{html.escape(target)}</h2>
<h3>Task definition and evaluation procedure</h3>{_report._procedure_grid_html(procedure)}
<h3>Task performance summary</h3>{_regression_performance_note(child.evaluation.protocol, 2000) if task == "regression" else _report._performance_methodology_html(child.evaluation.protocol, 2000)}{_report._html_table(performance, raw_html_cols=performance_raw)}
<h3>Data representations and learner impact on performance</h3>{_report._representation_impact_note(str(child.evaluation.optimize_metric), bool(getattr(child, "uses_modalities", False)))}{representation_fig if representation_fig else "<p>No representation-impact figure is available yet.</p>"}
<h3>Final MPMA-B specification</h3>{_report._html_table(mpma_b)}
<h3>Final MPMA-E specification</h3>{mpma_e_fig}{_report._html_table(selection)}{_report._html_table(members)}
<h3>{_report._top_mpma_heading(top_mpmas)}</h3>{_report._html_table(top_mpmas, raw_html_cols=top_raw)}
{oof}
<h3>Outer-unit strategy comparisons</h3>{_report._inferential_layer_html(child.evaluation.protocol, metric)}{_report._html_table(pairwise)}
{explainability_heading}<h4>Feature attribution</h4>{xai if xai else "<p>No explainability artefacts are available yet.</p>"}
{robustness_html}
{compute_heading}<p>Compute is summarized by additive CPU core-hours, model-fit count, and peak resident memory for the worker process tree. CPU time includes child processes and external R processes when used. MPMA-B and MPMA-E share the MPMA search pool, so their compute totals overlap.</p>{_report._html_table(compute)}
<h4>Hardware and runtime environment</h4>{_report._html_table(hardware_summary) if not hardware_summary.empty else "<p>Hardware details are unavailable for this run.</p>"}</section>'''


def _multilabel_summary(children: list[Sweep]) -> pd.DataFrame:

    rows = []

    for child in children:
        frame = _read_table(child.root() / "results" / "mpma_b_outer_results.parquet")

        if frame.empty:
            continue

        row: dict[str, Any] = {"target": str(child.data.target_col)}

        for metric in (
            "AUROC",
            "AUCPR",
            "AP",
            "MCC",
            "F1w",
            "BalAcc",
            "Accuracy",
            "log_loss",
            "brier",
        ):
            if metric in frame.columns:
                values = _finite(frame[metric])

                row[metric] = float(np.mean(values)) if len(values) else np.nan

        rows.append(row)

    label_summary = pd.DataFrame(rows)

    if label_summary.empty:
        return pd.DataFrame()

    numeric = [c for c in label_summary.columns if c != "target"]

    macro = pd.DataFrame(
        [
            {
                "Summary": "Macro across labels",
                **{
                    c: float(pd.to_numeric(label_summary[c], errors="coerce").mean())
                    for c in numeric
                },
            }
        ]
    )

    return macro


def _multilabel_oof_summary(
    children: list[Sweep], report_dir: Path
) -> tuple[pd.DataFrame, Path | None, pd.DataFrame, Path | None]:
    """Compute joint metrics from each label's held-out predictions."""
    truth: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    predictions: dict[str, pd.Series] = {}
    per_label: list[dict[str, Any]] = []
    for child in children:
        target = str(child.data.target_col)
        frame = _read_table(Path(child.root()) / "predictions" / "mpma_b_outer_predictions.parquet")
        required = {"outer_split_key", "sample_id", "y_true", "y_pred"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        if "y_proba_pos" in frame.columns:
            probability = pd.to_numeric(frame["y_proba_pos"], errors="coerce")
        else:
            positive_label = str(child.data.positive_class)
            col = f"proba_{positive_label}"
            if col not in frame.columns:
                continue
            probability = pd.to_numeric(frame[col], errors="coerce")
        repeat_ids = frame["outer_split_key"].map(_regression_repeat_id)
        keys = repeat_ids.astype(str) + "\0" + frame["sample_id"].astype(str)
        y = pd.to_numeric(frame["y_true"], errors="coerce")
        y_pred = pd.to_numeric(frame["y_pred"], errors="coerce")
        valid = y.notna() & probability.notna() & y_pred.notna()
        series_key = pd.Index(keys[valid], name="prediction_unit")
        if series_key.has_duplicates:
            raise ValueError(
                f"Multi-label predictions contain duplicate sample/repeat rows for {target!r}."
            )
        truth[target] = pd.Series(y[valid].to_numpy(dtype=int), index=series_key)
        scores[target] = pd.Series(probability[valid].to_numpy(dtype=float), index=series_key)
        predictions[target] = pd.Series(y_pred[valid].to_numpy(dtype=int), index=series_key)
        yt = truth[target].to_numpy(dtype=int)
        auc = roc_auc_score(yt, scores[target]) if len(np.unique(yt)) == 2 else np.nan
        ap = average_precision_score(yt, scores[target]) if len(np.unique(yt)) == 2 else np.nan
        per_label.append({"Target": target, "AUROC": auc, "Average precision": ap})
    if not truth:
        return pd.DataFrame(), None, pd.DataFrame(), None

    y_true = pd.concat(truth, axis=1).sort_index()
    y_score = pd.concat(scores, axis=1).reindex(index=y_true.index, columns=y_true.columns)
    y_predicted = pd.concat(predictions, axis=1).reindex(index=y_true.index, columns=y_true.columns)
    known = y_true.notna()
    flat_true = y_true.to_numpy(dtype=float)[known.to_numpy()].astype(int)
    flat_score = y_score.to_numpy(dtype=float)[known.to_numpy()]
    flat_pred = y_predicted.to_numpy(dtype=float)[known.to_numpy()].astype(int)
    fully_known = known.all(axis=1).to_numpy()
    exact_match = (
        accuracy_score(y_true.loc[fully_known].to_numpy(dtype=int),
                       y_predicted.loc[fully_known].to_numpy(dtype=int))
        if fully_known.any() else np.nan
    )
    auc_micro = roc_auc_score(flat_true, flat_score) if len(np.unique(flat_true)) == 2 else np.nan
    ap_micro = average_precision_score(flat_true, flat_score) if len(np.unique(flat_true)) == 2 else np.nan
    label_frame = pd.DataFrame(per_label)
    metrics = pd.DataFrame([{
        "Summary": "Joint held-out predictions (MPMA-B)",
        "AUROC micro": auc_micro,
        "AUROC macro": label_frame["AUROC"].mean(),
        "Average precision micro": ap_micro,
        "Average precision macro": label_frame["Average precision"].mean(),
        "F1 micro": f1_score(flat_true, flat_pred, zero_division=0),
        "F1 macro": f1_score(flat_true, flat_pred, average="macro", zero_division=0),
        "Hamming loss": hamming_loss(flat_true, flat_pred),
        "Exact match": exact_match,
        "Observed label predictions": int(known.to_numpy().sum()),
        "Fully observed prediction rows": int(fully_known.sum()),
    }])
    figure_path = report_dir / "figures" / "multilabel_performance.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_data = label_frame.set_index("Target")[["AUROC", "Average precision"]]
    ax = plot_data.plot(kind="bar", figsize=(9, 4.5), ylim=(0, 1), rot=30)
    ax.set_ylabel("Score")
    ax.set_title("Per-diagnosis held-out discrimination (MPMA-B)")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_path, dpi=160)
    plt.close(ax.figure)

    patient_detection, control_accuracy, detection_figure, control_figure = (
        _multilabel_patient_detection(children, report_dir)
    )
    return (
        metrics,
        figure_path,
        patient_detection,
        detection_figure,
        control_accuracy,
        control_figure,
    )


def _multilabel_patient_detection(
    children: list[Sweep], report_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, Path | None]:
    """Summarize detection of present labels and all-absent healthy controls."""
    detection_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for procedure, relative_path in (
        ("MPMA-B selected single model", "predictions/mpma_b_outer_predictions.parquet"),
        ("MPMA-E selected ensemble", "ensembling/ensemble_predictions.parquet"),
    ):
        truth_by_target: dict[str, pd.Series] = {}
        prediction_by_target: dict[str, pd.Series] = {}
        for child in children:
            target = str(child.data.target_col)
            frame = _read_table(Path(child.root()) / relative_path)
            required = {"sample_id", "y_true", "y_pred"}
            split_col = "outer_split_key" if "outer_split_key" in frame else "split_key"
            if frame.empty or not required.issubset(frame.columns) or split_col not in frame:
                continue
            repeat_ids = frame[split_col].map(_regression_repeat_id)
            keys = repeat_ids.astype(str) + "\0" + frame["sample_id"].astype(str)
            y = pd.to_numeric(frame["y_true"], errors="coerce")
            y_pred = pd.to_numeric(frame["y_pred"], errors="coerce")
            valid = y.notna() & y_pred.notna()
            series_key = pd.Index(keys[valid], name="prediction_unit")
            if series_key.has_duplicates:
                raise ValueError(
                    f"{procedure} predictions contain duplicate sample/repeat rows for {target!r}."
                )
            truth_by_target[target] = pd.Series(
                y[valid].to_numpy(dtype=int), index=series_key
            )
            prediction_by_target[target] = pd.Series(
                y_pred[valid].to_numpy(dtype=int), index=series_key
            )
        if not truth_by_target:
            continue

        y_true = pd.concat(truth_by_target, axis=1).sort_index()
        y_predicted = pd.concat(prediction_by_target, axis=1).reindex(
            index=y_true.index, columns=y_true.columns
        )
        known = y_true.notna()
        positive_target = str(children[0].data.positive_class)
        class_labels = tuple(str(value) for value in children[0].data.class_labels)
        if positive_target not in class_labels or len(class_labels) != 2:
            raise ValueError(
                "Patient-level positive-label summaries require binary class_labels "
                "containing positive_class."
            )
        positive_value = class_labels.index(positive_target)
        negative_value = next(i for i in range(2) if i != positive_value)
        actual_present = y_true.eq(positive_value) & known
        predicted_present = y_predicted.eq(positive_value)
        n_present = actual_present.sum(axis=1).astype(int)
        n_detected = (actual_present & predicted_present).sum(axis=1).astype(int)
        repeat = pd.Series(
            [str(key).split(chr(0), 1)[0] for key in y_true.index],
            index=y_true.index,
        )
        patient_rows = pd.DataFrame(
            {
                "repeat": repeat,
                "detected": n_detected,
                "present": n_present,
            }
        )
        patient_rows = patient_rows.loc[patient_rows["present"] > 0].copy()
        patient_rows["category"] = (
            patient_rows["detected"].astype(str)
            + "/"
            + patient_rows["present"].astype(str)
            + " Present diagnoses detected"
        )
        if not patient_rows.empty:
            counts = patient_rows.groupby(["category", "repeat"], observed=True).size()
            totals = patient_rows.groupby("repeat").size()
            for (category, repeat_id), count in counts.items():
                detection_frames.append(
                    pd.DataFrame([{
                        "Prediction procedure": procedure,
                        "CV repeat": repeat_id,
                        "Present diagnoses detected / observed": category,
                        "Patient-repeat rows": int(count),
                        "Positive patients (%)": float(count / totals.loc[repeat_id] * 100),
                    }])
                )

        confirmed_control = known.all(axis=1) & y_true.eq(negative_value).all(axis=1)
        n_controls = int(confirmed_control.sum())
        if n_controls:
            correct_controls = (
                y_predicted.loc[confirmed_control].eq(negative_value).all(axis=1)
            )
            repeat_control = repeat.loc[confirmed_control]
            for repeat_id, indices in repeat_control.groupby(repeat_control).groups.items():
                correct_count = int(correct_controls.loc[indices].sum())
                total_count = int(len(indices))
                control_frames.append(
                    pd.DataFrame([{
                        "Prediction procedure": procedure,
                        "CV repeat": repeat_id,
                        "Confirmed healthy controls": total_count,
                        "All diagnoses predicted Absent": correct_count,
                        "Healthy controls correctly identified (%)": float(
                            correct_count / total_count * 100
                        ),
                    }])
                )

    detection = (
        pd.concat(detection_frames, ignore_index=True)
        if detection_frames else pd.DataFrame()
    )
    controls = (
        pd.concat(control_frames, ignore_index=True)
        if control_frames else pd.DataFrame()
    )
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    detection_figure = figure_dir / "multilabel_patient_positive_detection.png"
    control_figure = figure_dir / "multilabel_healthy_control_exact_match.png"
    if not detection.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot = detection.pivot_table(
            index="Present diagnoses detected / observed",
            columns=["Prediction procedure", "CV repeat"],
            values="Positive patients (%)",
            fill_value=0,
        )
        ax = plot.plot(kind="bar", figsize=(11, 5), ylim=(0, 100), rot=35)
        ax.set_ylabel("Positive patient-repeat rows (%)")
        ax.set_xlabel("Present diagnoses correctly detected / observed Present diagnoses")
        ax.set_title("Patient-level detection of Present diagnoses")
        ax.legend(title="Procedure / CV repeat", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.figure.tight_layout()
        ax.figure.savefig(detection_figure, dpi=160, bbox_inches="tight")
        plt.close(ax.figure)
    else:
        detection_figure = None
    if not controls.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot = controls.pivot_table(
            index="Prediction procedure",
            columns="CV repeat",
            values="Healthy controls correctly identified (%)",
            fill_value=0,
        )
        ax = plot.plot(kind="bar", figsize=(8, 4.5), ylim=(0, 100), rot=0)
        ax.set_ylabel("Healthy controls with all diagnoses predicted Absent (%)")
        ax.set_xlabel("")
        ax.set_title("Exact identification of confirmed healthy controls")
        ax.legend(title="CV repeat")
        ax.figure.tight_layout()
        ax.figure.savefig(control_figure, dpi=160)
        plt.close(ax.figure)
    else:
        control_figure = None
    return detection, controls, detection_figure, control_figure


def write_multi_target_report(
    sweep: Sweep, children: list[Sweep] | None = None
) -> dict[str, Path]:

    children = target_sweeps(sweep) if children is None else list(children)

    root = Path(sweep.root())

    report_dir = root / "report"

    report_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = report_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    stage("Report", str(report_dir))

    with phase_progress("Combined report", 3) as phase:
        phase.phase("target overview")

        overview = pd.DataFrame([_target_primary_summary(child) for child in children])

        phase.phase("task summaries")

        task = _normalise_sweep_task(sweep.data.task)
        combined_task_label = (
            "Multilabel"
            if task == "multilabel"
            else "Multi-target regression"
            if children
            and all(
                _target_task(child.data, str(child.data.target_col)) == "regression"
                for child in children
            )
            else "Heterogeneous multi-output"
        )

        macro = (
            _multilabel_summary(children) if task == "multilabel" else pd.DataFrame()
        )
        (
            joint_metrics,
            joint_figure,
            patient_detection,
            patient_detection_figure,
            healthy_control_accuracy,
            healthy_control_figure,
        ) = (
            _multilabel_oof_summary(children, report_dir)
            if task == "multilabel"
            else (pd.DataFrame(), None, pd.DataFrame(), None, pd.DataFrame(), None)
        )
        pca_stability = _pca_alignment_stability_table(children)
        pca_example = _pca_example_subject_predictions(children, "SYN0001")
        pca_spec = getattr(sweep.data, "fold_pca_target", None)
        pca_example_figure = (
            _write_pca_example_radar(pca_example, report_dir, list(pca_spec.domains))
            if pca_spec is not None
            else None
        )

        procedure = pd.DataFrame(
            [
                ["Task", sweep.title],
                [
                    "Task type",
                    combined_task_label,
                ],
                [
                    "Targets",
                    ", ".join(str(child.data.target_col) for child in children),
                ],
                ["Target count", len(children)],
                ["Data format", _task_data_format(sweep)],
                ["Procedure", sweep.evaluation.protocol],
                ["Benchmark", sweep.evaluation.benchmark_id or ""],
                ["Outer folds", sweep.evaluation.outer_folds],
                ["Inner folds", sweep.evaluation.inner_folds],
                ["Repeats", sweep.evaluation.repeats],
                ["Random seed", sweep.evaluation.random_state],
                [
                    "Selection rule",
                    "Reported performance uses held-out outer evaluation predictions.",
                ],
            ],
            columns=["Field", "Value"],
        )
        if pca_spec is not None:
            procedure.loc[len(procedure)] = [
                "Target construction",
                _fold_pca_target_description(pca_spec),
            ]

        phase.phase("target report sections")

        sections = "".join(
            _target_report_section(child, report_dir, include_nav_landmarks=index == 0)
            for index, child in enumerate(children)
        )

    css = _report._report_css()

    joint_html = ""
    if not joint_metrics.empty and joint_figure:
        joint_html = (
            '<h2>Joint multi-label performance</h2>'
            '<p>Metrics use aligned held-out MPMA-B predictions. Missing diagnosis labels are omitted from cell-wise metrics; '
            'exact match is calculated only for rows with every diagnosis observed.</p>'
            f'{_report._html_table(joint_metrics)}'
            f'<img alt="Per-diagnosis held-out performance" style="max-width:100%;height:auto" '
            f'src="report/figures/{joint_figure.name}">'
        )
        if not patient_detection.empty and patient_detection_figure:
            joint_html += (
                '<h3>Patient-level detection of Present diagnoses</h3>'
                '<p>For each patient with at least one observed Present diagnosis, this shows how many of those '
                'Present labels were predicted Present. Missing labels are excluded. MPMA-B uses the single '
                'model selected within each outer fold; MPMA-E uses the ensemble selected within each outer fold, '
                'with selection based on inner validation predictions.</p>'
                f'{_report._html_table(patient_detection)}'
                f'<img alt="Patient-level detection of Present diagnoses" style="max-width:100%;height:auto" '
                f'src="report/figures/{patient_detection_figure.name}">'
            )
        if not healthy_control_accuracy.empty and healthy_control_figure:
            joint_html += (
                '<h3>Healthy control exact-match performance</h3>'
                '<p>A confirmed healthy control has all diagnosis labels observed and every true label is Absent. '
                'The table reports how many such patients each procedure predicted Absent for every diagnosis.</p>'
                f'{_report._html_table(healthy_control_accuracy)}'
                f'<img alt="Healthy control exact-match performance" style="max-width:100%;height:auto" '
                f'src="report/figures/{healthy_control_figure.name}">'
            )

    pca_html = ""
    if pca_spec is not None:
        if not pca_stability.empty:
            pca_html += (
                '<h2 id="pca-alignment-stability">PCA alignment and stability</h2>'
                '<p>Each row summarizes the selected MPMA-B component match across outer folds. '
                'Absolute Tucker congruence measures loading-pattern similarity to the corresponding '
                'reference component; values at or above the configured threshold indicate adequate '
                'alignment. Match consistency reports how often the same pre-alignment fold component '
                'was assigned to that reference component; component numbers are fold-local bookkeeping, '
                'so congruence is the primary stability measure. Sign adjustments orient fold scores to '
                'the reference direction.</p>'
                f'{_report._html_table(pca_stability)}'
            )
        else:
            pca_html += (
                '<h2 id="pca-alignment-stability">PCA alignment and stability</h2>'
                '<p>Alignment diagnostics will appear here after selected outer-fold results are available.</p>'
            )
        if pca_example_figure is not None:
            pca_html += (
                '<h2 id="pca-example-subject">Example subject scores</h2>'
                '<p>The radar plot shows SYN0001’s true fold-specific scores and selected MPMA-B '
                'predictions for each domain. Values are averaged over the subject’s held-out appearances '
                'across CV repeats; true scores are generated by the PCA fitted on each corresponding '
                'outer-training fold. The radial axis is shifted uniformly for display; tick labels show '
                'the original score values.</p>'
                f'<img alt="SYN0001 true and predicted PCA domain scores" style="max-width:100%;height:auto" '
                f'src="report/figures/{pca_example_figure.name}">'
            )
        else:
            pca_html += (
                '<h2 id="pca-example-subject">Example subject scores</h2>'
                '<p>The SYN0001 radar plot will appear after MPMA-B held-out predictions are available for all four domains.</p>'
            )
    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>mllabiome report</title><link rel="icon" type="image/svg+xml" href="{_report._favicon_href()}"><style>{css}</style></head>
<body>{_report._report_nav_html()}
<div class="report-shell"><main id="top" class="report-content"><h2 id="performance-evaluation" class="first-section">Task definition and evaluation procedure</h2>{_report._procedure_grid_html(procedure)}
<h2 id="performance-summary">Target performance overview</h2><p>Every target retains its task-appropriate primary metric. Values are mean ± SD across {"held-out datasets" if is_lodo_protocol(sweep.evaluation.protocol) else "outer test folds"}. Classification and regression metrics are intentionally not collapsed into a single heterogeneous score.</p>{_report._html_table(overview)}{_report._html_table(macro) if not macro.empty else ""}
{pca_html}
{joint_html}
{sections}{_report._report_footer_html()}</main></div></body></html>'''

    html_text = _report._sanitize_report_html(html_text)

    html_path = prepare_report_html_path(root)

    html_path.write_text(html_text, encoding="utf-8")

    important_frames: list[pd.DataFrame] = []

    hardware_frames: list[pd.DataFrame] = []

    for child in children:
        target = str(child.data.target_col)

        task_name = _target_task(child.data, target)

        child_features = _read_table(
            Path(child.root()) / "report" / "tables" / "important_features.parquet"
        )

        if not child_features.empty:
            child_features = child_features.copy()

            child_features.insert(0, "Task", task_name)

            child_features.insert(0, "Target", target)

            important_frames.append(child_features)

        child_hardware = _read_table(
            Path(child.root()) / "report" / "tables" / "hardware_environment.parquet"
        )

        if not child_hardware.empty:
            child_hardware = child_hardware.copy()

            child_hardware.insert(0, "Task", task_name)

            child_hardware.insert(0, "Target", target)

            hardware_frames.append(child_hardware)

    important_features_path = tables_dir / "important_features.parquet"

    hardware_environment_path = tables_dir / "hardware_environment.parquet"

    pca_stability_path = tables_dir / "pca_alignment_stability.parquet"
    pca_example_predictions_path = tables_dir / "pca_example_subject_predictions.parquet"

    patient_detection_path = tables_dir / "multilabel_patient_positive_detection.parquet"

    healthy_control_path = tables_dir / "multilabel_healthy_control_exact_match.parquet"

    write_table(
        important_features_path,
        pd.concat(important_frames, ignore_index=True)
        if important_frames
        else pd.DataFrame(),
    )

    write_table(
        hardware_environment_path,
        pd.concat(hardware_frames, ignore_index=True)
        if hardware_frames
        else pd.DataFrame(),
    )

    if pca_spec is not None:
        write_table(pca_stability_path, pca_stability)
        if not pca_example.empty:
            write_table(pca_example_predictions_path, pca_example)

    if task == "multilabel":
        write_table(patient_detection_path, patient_detection)

        write_table(healthy_control_path, healthy_control_accuracy)

    manifest_path = report_dir / "report_manifest.json"

    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": task,
            "report_template": _report.REPORT_TEMPLATE_VERSION,
            "report_layers": list(_report.REPORT_LAYERS),
            "targets": [
                {
                    "target": str(child.data.target_col),
                    "task": _target_task(child.data, str(child.data.target_col)),
                }
                for child in children
            ],
            "pca_alignment_stability": (
                pca_stability_path if pca_spec is not None else None
            ),
            "pca_example_subject_radar": (
                pca_example_figure if pca_example_figure is not None else None
            ),
            "root_report": html_path,
        },
        manifest_path,
    )

    outputs = {
        "html_report": html_path,
        "manifest": manifest_path,
        "important_features": important_features_path,
        "hardware_environment": hardware_environment_path,
    }
    if pca_spec is not None:
        outputs["pca_alignment_stability"] = pca_stability_path
        if pca_example_figure is not None:
            outputs["pca_example_subject_radar"] = pca_example_figure
            outputs["pca_example_subject_predictions"] = pca_example_predictions_path
    if task == "multilabel" and not patient_detection.empty:
        outputs["patient_positive_detection"] = patient_detection_path
    if task == "multilabel" and not healthy_control_accuracy.empty:
        outputs["healthy_control_exact_match"] = healthy_control_path

    stage("Combined run summary", str(root))

    _report._terminal_table(
        "Task definition and evaluation procedure",
        _report._compact_procedure_for_terminal(procedure),
    )

    _report._terminal_table("Target performance overview", overview)

    if not macro.empty:
        _report._terminal_table("Macro target summary", macro)
    if not joint_metrics.empty:
        _report._terminal_table("Joint multi-label performance", joint_metrics)

    for child in children:
        child_root = Path(child.root())

        summary = _read_json(child_root / "run_summary.json")

        machine = (
            summary.get("machine", {})
            if isinstance(summary.get("machine"), dict)
            else {}
        )

        environment = {
            **machine,
            "evaluation_workers": summary.get("workers"),
            "threads_per_worker": summary.get("threads_per_worker"),
        }

        hardware = _report._hardware_summary_table(environment)

        if not hardware.empty:
            _report._terminal_table(
                f"Hardware and runtime · {child.data.target_col}", hardware
            )

    success("Report completed")

    path_table("Report outputs", outputs)

    return outputs


def write_task_report(sweep: Sweep) -> dict[str, Any]:

    children = target_sweeps(sweep)

    if len(children) > 1 or children[0] is not sweep:
        target_outputs: dict[str, Any] = {}

        for index, child in enumerate(children, start=1):
            info(f"Report · target {index}/{len(children)} · {child.data.target_col}")

            if (child.root() / "ensembling" / "selected_unit.json").exists():
                build_final_models(child.root())

            task = _target_task(child.data, str(child.data.target_col))

            target_outputs[str(child.data.target_col)] = (
                write_regression_report(child, emit_html=False)
                if task == "regression"
                else write_report(child, emit_html=False)
            )

        combined = write_multi_target_report(sweep, children)

        return {"targets": target_outputs, "combined": combined}

    child = children[0]

    if (child.root() / "ensembling" / "selected_unit.json").exists():
        build_final_models(child.root())

    task = _target_task(child.data, str(child.data.target_col))

    return (
        write_regression_report(child) if task == "regression" else write_report(child)
    )
