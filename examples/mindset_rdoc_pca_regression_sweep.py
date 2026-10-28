"""Nested-CV regression of fold-local PCA domain scores from synthetic microbiome data."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from mllabiome import mll


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TITLE = "MINDset fold-local PCA domain-score regression sweep"
EXPERIMENT_DIR = HERE / "runs" / "Mindset-rdoc-indepedent-targets-regression"

METADATA = mll.Metadata(
    metadata_path=HERE / "data" / "MINDset_simulation" / "simulated_metadata_no_signal.tsv",
)

QUESTIONNAIRE_COLUMNS = (
    "ASI_Tot",
    "AQ_AttDetail",
    "AQ_Socialskill",
    "AQ_Communication",
    "AQ_AttSwitch",
    "AQ_Imagination",
    "BRIEFA_EmoContr_R",
    "BRIEFA_Inhib_R",
    "BRIEFA_Initiate_R",
    "BRIEFA_OrgMat_R",
    "BRIEFA_PlanOrg_R",
    "BRIEFA_SelfMon_R",
    "BRIEFA_Shift_R",
    "BRIEFA_TaskMon_R",
    "BRIEFA_WorkMem_R",
    "CAARS_SeCo",
    "CAARS_InMe",
    "CAARS_ImEm",
    "CAARS_HyRe",
    "IDS_Tot_Intake",
    "ptq_CC",
    "ptq_U",
    "ptq_CMC",
    "PID_R_Anta",
    "PID_R_Det",
    "PID_R_Dis",
    "PID_R_Neg",
    "PID_R_Psy",
    "tt_DIF",
    "tt_DDF",
    "tt_EOT",
)

# The reference PCA was fitted on the 324 questionnaire-only participants.
# These labels and signs are fixed before evaluating the 173 synthetic samples.
DOMAIN_TO_COMPONENT = (
    ("Negative_Valence", "PC1"),
    ("Cognitive_Systems", "PC2"),
    ("Arousal_Regulatory", "PC3"),
    ("Social_Processes", "PC4"),
)
REFERENCE_LOADINGS = (
    PROJECT_ROOT
    / "data"
    / "reference_factor_loadings.csv"
)
FOLD_PCA_TARGET = mll.FoldPCATargetSpec(
    questionnaire_columns=QUESTIONNAIRE_COLUMNS,
    reference_loadings_path=REFERENCE_LOADINGS,
    domain_to_component=DOMAIN_TO_COMPONENT,
    n_components=4,
    promax_power=4,
    min_alignment_congruence=0.85,
    reference_n=324,
)

DOMAINS = tuple(domain for domain, _ in DOMAIN_TO_COMPONENT)
DATA = mll.Data(
    abundance_path=HERE / "data" / "MINDset_simulation" / "simulated_biome_strongsignal_all.tsv",
    metadata=METADATA,
    format="metaphlan_tsv",
    sample_id_col="Deelnemerscode",
    subject_id_col="Deelnemerscode",
    target_col=DOMAINS,
    task="regression",
    target_tasks={domain: "regression" for domain in DOMAINS},
    fold_pca_target=FOLD_PCA_TARGET,
)

RESOLUTIONS = (
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("family-genus", ("family", "genus")),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("relative_abundance", composition_scope="joint"),
    mll.Transformation("hellinger", composition_scope="joint"),
    mll.Transformation("clr", composition_scope="joint"),
)

# replacing classifiers with their regression counterparts.
MODELS = (
    (
        "RF_1000_msl5",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "ExtraTrees_1000_msl5",
        ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features=1.0,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "XGB_250_lr0.1",
        XGBRegressor(
            learning_rate=0.1,
            n_estimators=250,
            max_depth=3,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            random_state=42,
        ),
    ),
    ("Ridge_Base", Ridge(alpha=1.0)),
)

EVALUATION = mll.Evaluation(
    optimize_metric="rmse",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
    n_jobs=2,
)

GATE = mll.QualificationGate(enabled=False)

ENSEMBLE = mll.Ensemble(
    include_inactive=True,
    max_sizes=(3, 5, 10),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=(
        "mean_prediction",
        "weighted_mean_prediction",
        "median_prediction",
    ),
    optimize_metric="rmse",
)

SWEEP = mll.Sweep(
    data=DATA,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    resolutions=RESOLUTIONS,
    count_transformations=COUNT_TRANSFORMATIONS,
    learners=MODELS,
    evaluation=EVALUATION,
    gate=GATE,
    ensemble=ENSEMBLE,
)

