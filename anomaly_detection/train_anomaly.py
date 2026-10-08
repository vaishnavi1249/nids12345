import os

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest

from anomaly_detector import AnomalyDetector


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    "dataset",
    "images",
    "train.npz"
)

MODEL_PATH = os.path.join(
    "models",
    "anomaly_detector.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_NORMAL_SAMPLES = 5000

CONTAMINATION = 0.01

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print(
    "[*] Loading training dataset..."
)


if not os.path.exists(
    DATASET_PATH
):

    raise FileNotFoundError(
        f"Dataset not found: "
        f"{DATASET_PATH}"
    )


with np.load(
    DATASET_PATH
) as data:

    X = data["X"]

    y = data["y"]


print(
    f"[+] X shape: {X.shape}"
)

print(
    f"[+] y shape: {y.shape}"
)


# ============================================================
# SELECT ONLY NORMAL TRAFFIC
# ============================================================

normal_mask = (
    y == 0
)

normal_images = (
    X[normal_mask]
)


if len(normal_images) == 0:

    raise RuntimeError(
        "No Normal samples found "
        "in train.npz."
    )


# Limit training size.
if (
    len(normal_images)
    >
    MAX_NORMAL_SAMPLES
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(normal_images),
        size=MAX_NORMAL_SAMPLES,
        replace=False
    )

    normal_images = (
        normal_images[indices]
    )


print(
    f"[+] Using "
    f"{len(normal_images)} "
    "normal images."
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print(
    "[*] Extracting anomaly features..."
)


features = np.asarray(
    [
        AnomalyDetector.extract_features(
            image
        )
        for image in normal_images
    ],
    dtype=np.float32
)


print(
    f"[+] Feature matrix: "
    f"{features.shape}"
)


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print(
    "[*] Training Isolation Forest..."
)


model = IsolationForest(

    n_estimators=200,

    contamination=CONTAMINATION,

    random_state=RANDOM_STATE,

    n_jobs=-1,
)


model.fit(
    features
)


# ============================================================
# LEARN NORMALITY THRESHOLD
# ============================================================

normal_scores = (
    model.decision_function(
        features
    )
)


threshold = float(
    np.percentile(
        normal_scores,
        CONTAMINATION * 100.0
    )
)


# ============================================================
# SAVE
# ============================================================

bundle = {

    "model":
        model,

    "threshold":
        threshold,

    "feature_version":
        "v1",

    "training_samples":
        int(
            len(normal_images)
        ),

    "contamination":
        CONTAMINATION,
}


os.makedirs(
    os.path.dirname(
        MODEL_PATH
    ),
    exist_ok=True
)


joblib.dump(
    bundle,
    MODEL_PATH
)


print()
print(
    "[+] Anomaly detector saved:"
)

print(
    f"    {MODEL_PATH}"
)

print(
    f"[+] Learned threshold: "
    f"{threshold:.6f}"
)

print(
    "[+] Anomaly training complete."
)