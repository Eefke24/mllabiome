from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "Mindset Multi-Label Microbiome Sweep"
EXPERIMENT_DIR = HERE / "runs" / "Mindset-Simulation"

METADATA = mll.Metadata(
    metadata_path=HERE / "data" / "MINDset_simulation" / "mllabiome_metadata.tsv",
)

DATA = mll.Data(
    abundance_path=HERE / "data" / "MINDset_simulation" / "mllabiome_simulation_data_exclusive.tsv",
    metadata=METADATA,
    format="metaphlan_tsv",
    sample_id_col="sampleID", 
    target_col = "Mood",
    task = "classification",
    class_labels=("0", "1"),
    positive_class="1",
    subject_id_col="sampleID",
    # target_col=("Mood", "ASD", "ADHD", "Anxiety", "SUD"),
    # task="multilabel_classification", 
)

# 3. EXPLORE CONFIGURATION
EXPLORE = mll.Explore(
    ranks=("family", "genus"), # Added family and genus as requested
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
)

# (multi-label compatible)
# MODELS = (
#     (
#         "RF_1000_msl5",
#         RandomForestClassifier(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             n_jobs=1,
#             random_state=42,
#         ),
#     ),
#     (
#         "ExtraTrees_1000_msl5",
#         ExtraTreesClassifier(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             max_features="sqrt",
#             n_jobs=1,
#             random_state=42,
#         ),
#     ),
#     (
#         # Wrapping XGBoost to handle multi-label outputs natively
#         "XGB_Multi_250_lr0.1",
#         MultiOutputClassifier(
#             XGBClassifier(
#                 learning_rate=0.1,
#                 n_estimators=250,
#                 max_depth=3,
#                 objective="binary:logistic",
#                 eval_metric="logloss",
#                 n_jobs=1,
#                 random_state=42,
#             )
#         )
#     ),
# )


# EVALUATION = mll.Evaluation.benchmark(
#     optimize_metric="log_loss",
#     n_jobs="auto",
# )

EVALUATION = mll.Evaluation(
    optimize_metric="log_loss",
    outer_folds=2,      # Minimal folds for a fast test
    inner_folds=2,      # Minimal inner folds
    repeats=1,          # No repeating for the test run
    n_jobs=1            # Set to 1 to easily read terminal errors if they happen
)


GATE = mll.QualificationGate(
    enabled=False,
)


ENSEMBLE = mll.Ensemble(
    max_sizes=(3,),
    selection_strategies=(
        "top_k",
        #"best_per_resolution",
        #"best_per_learner_type",
    ),
    aggregation_strategies=(
        "mean_proba",
        #"weighted_mean_proba",
    ),
    optimize_metric="log_loss",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b","mpma_e"),
    profile="screening",
    methods=(
        mll.Permutation(),
        mll.SHAP(),
    ),
    classes="auto",
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b","mpma_e"),
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