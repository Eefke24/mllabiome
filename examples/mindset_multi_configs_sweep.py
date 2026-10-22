from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "Mindset independent multi-label microbiome sweep"
EXPERIMENT_DIR = HERE / "runs" / "Mindset-multilabel-independent-targets"

METADATA = mll.Metadata(
    metadata_path=HERE / "data" / "MINDset_simulation" / "simulated_metadata_no_signal.tsv",
)

DATA = mll.Data(
    abundance_path=HERE / "data" / "MINDset_simulation" / "simulated_biome_strongsignal_all.tsv",
    metadata=METADATA,
    format="metaphlan_tsv", 
    sample_id_col="Deelnemerscode",
    subject_id_col="Deelnemerscode",
    target_col=("MoodDisFinal", "ASD_Final", "ADHD_Final", "AnxDisFinal", "VerslavingFinal"),
    task="multilabel",
    class_labels=("Absent", "Present"),
    positive_class="Present",
)

# 3. EXPLORE CONFIGURATION
EXPLORE = mll.Explore(
    ranks=("family", "genus"), 
    top_taxa=12,
    heatmap_top=30,
    min_prevalence=0.10,
    differential_abundance="off",
    detection_limit=0.0,
    permutations=999,
    bootstrap_replicates=2000,
    confidence_level=0.95,
    random_state=42,
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

MODELS = (
    (
        "RF_1000_msl5",
        RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "ExtraTrees_1000_msl5",
        ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "XGB_250_lr0.1",
        XGBClassifier(
            learning_rate=0.1,
            n_estimators=250,
            max_depth=3,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )
    ),
    (
        "Ridge_Base",
        RidgeClassifier(
            class_weight="balanced",
            random_state=42,
        )
    ),
    (
        "LogisticRegression_ElasticNet",
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            random_state=42,
        ),
    ),
    (
        "LinearSVC_Calibrated",
        mll.build_learner("calib_lsvc"),
    ),
    (
        "RBF_SVC_Calibrated",
        mll.build_learner("svc_rbf"),
    ),
    (
        "KumoTabular_small",
        mll.KumoTabularClassifier(
        size="small",
        device="cpu",
        random_state=42,
        ),
    ),
)


# EVALUATION = mll.Evaluation.benchmark(
#     optimize_metric="log_loss",
#     n_jobs="auto",
# )

EVALUATION = mll.Evaluation(
    optimize_metric="log_loss",
    outer_folds=5,      
    inner_folds=3,      
    repeats=2,          
    n_jobs=2            
)


GATE = mll.QualificationGate(
    enabled=False,
)

ENSEMBLE = mll.Ensemble(
    include_inactive=True,
    max_sizes=(3, 5, 10,),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "performance_diversity",
        "caruana",
        "bagged_caruana",
        "super_learner",
        "safe_super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "logistic_stacking",
        "rf_stacking",
    ),
    optimize_metric="log_loss",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(
        mll.Permutation(),
        mll.ALE(),
        mll.LIME(),
    ),
    classes="auto",
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b",),
    top_k=30,
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
    explore=EXPLORE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    robustness=ROBUSTNESS,
)
