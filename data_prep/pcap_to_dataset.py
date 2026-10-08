"""
CIC-IDS2017 CSV -> 9-Class RGB Image Dataset

CSV-only dataset builder.

This version intentionally does NOT:
- read PCAP files
- perform PCAP/CSV timestamp calibration
- match flows back to PCAP packets

It directly uses the labelled CIC-IDS2017 flow CSV files.

Output:
    dataset/images_final/
        train.npz
        val.npz
        test.npz

Each sample:
    shape = (9, 1486, 3)

Classes:
    0 Normal
    1 DoS
    2 DDoS
    3 Port Scan
    4 Brute Force
    5 Botnet
    6 Web Attack
    7 Infiltration
    8 Heartbleed
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# PROJECT CLASSES
# ============================================================

CLASS_NAMES = [
    "Normal",
    "DoS",
    "DDoS",
    "Port Scan",
    "Brute Force",
    "Botnet",
    "Web Attack",
    "Infiltration",
    "Heartbleed",
]

NUM_CLASSES = len(CLASS_NAMES)

# CNN input expected by the existing project.
IMAGE_HEIGHT = 9
IMAGE_WIDTH = 1486
IMAGE_CHANNELS = 3

IMAGE_SHAPE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {
    # Normal
    "normal": 0,
    "benign": 0,

    # DoS
    "dos": 1,
    "dos hulk": 1,
    "dos goldeneye": 1,
    "dos slowloris": 1,
    "dos slowhttptest": 1,

    # DDoS
    "ddos": 2,

    # Port Scan
    "portscan": 3,
    "port scan": 3,

    # Brute Force
    "ftp-patator": 4,
    "ssh-patator": 4,
    "brute force": 4,

    # Botnet
    "bot": 5,
    "botnet": 5,

    # Web Attack
    "web attack brute force": 6,
    "web attack xss": 6,
    "web attack sql injection": 6,

    # Infiltration
    "infiltration": 7,

    # Heartbleed
    "heartbleed": 8,
}


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):
    if label is None:
        return ""

    text = str(label)

    # Remove weird CIC encoding variants.
    text = (
        text.replace("\x96", "-")
        .replace("\x97", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    text = " ".join(
        text.strip().lower().split()
    )

    return text


def map_label(label):
    key = normalize_label(label)

    if key in LABEL_MAP:
        return LABEL_MAP[key]

    # Tolerant Web Attack handling.
    if key.startswith("web attack"):
        if "brute force" in key:
            return 6

        if "xss" in key:
            return 6

        if "sql injection" in key:
            return 6

    # Tolerant DoS handling.
    if key.startswith("dos "):
        return 1

    return None


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):
    for column in df.columns:
        clean = str(column).strip().lower()

        if clean == "label":
            return column

    raise ValueError(
        "Could not find 'Label' column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

def clean_feature_dataframe(df):
    """
    Keep only numeric flow features.

    Removes:
        Flow ID
        Source IP
        Destination IP
        Timestamp
        Label

    Handles:
        NaN
        +Infinity
        -Infinity
        duplicated feature columns
    """

    work = df.copy()

    # Clean column names.
    work.columns = [
        str(column).strip()
        for column in work.columns
    ]

    label_column = find_label_column(work)

    # Preserve labels separately.
    raw_labels = work[label_column].copy()

    # Remove non-feature columns.
    remove_columns = []

    for column in work.columns:
        clean = str(column).strip().lower()

        if clean in {
            "flow id",
            "source ip",
            "destination ip",
            "timestamp",
            "src ip",
            "dst ip",
        }:
            remove_columns.append(column)

    remove_columns.append(label_column)

    feature_df = work.drop(
        columns=list(
            dict.fromkeys(remove_columns)
        ),
        errors="ignore",
    )

    # Convert everything remaining to numeric.
    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove columns that became completely empty.
    feature_df = feature_df.dropna(
        axis=1,
        how="all",
    )

    # Replace infinite values.
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill missing values using column medians.
    for column in feature_df.columns:

        values = feature_df[column]

        median = values.median()

        if pd.isna(median):
            median = 0.0

        feature_df[column] = values.fillna(
            median
        )

    # Final safety conversion.
    feature_df = feature_df.astype(
        np.float32
    )

    return feature_df, raw_labels


# ============================================================
# CSV LOADING
# ============================================================

def load_csv_file(csv_path):
    print()
    print("=" * 70)
    print(f"[*] Loading: {csv_path}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        encoding="latin1",
        low_memory=False,
    )

    print(
        f"[*] Rows loaded: {len(df):,}"
    )

    print(
        f"[*] Columns: {len(df.columns)}"
    )

    feature_df, raw_labels = (
        clean_feature_dataframe(df)
    )

    labels = []

    for label in raw_labels:
        labels.append(
            map_label(label)
        )

    labels = np.array(
    [
        -1 if label is None else int(label)
        for label in labels
    ],
    dtype=np.int64,
)

    valid_mask = labels >= 0

    feature_df = feature_df.loc[
        valid_mask
    ].reset_index(drop=True)

    labels = labels[
        valid_mask
    ]

    print(
        f"[*] Usable labelled rows: "
        f"{len(labels):,}"
    )

    print(
        f"[*] Numeric features: "
        f"{feature_df.shape[1]}"
    )

    distribution = Counter(
        labels.tolist()
    )

    for class_id in range(NUM_CLASSES):
        print(
            f"    {CLASS_NAMES[class_id]:15s}: "
            f"{distribution.get(class_id, 0):,}"
        )

    return feature_df, labels


# ============================================================
# FEATURE MATRIX -> RGB IMAGE
# ============================================================

def features_to_image(
    feature_vector,
    feature_min,
    feature_max,
):
    """
    Convert one normalized flow feature vector
    into a deterministic 9 x 1486 x 3 RGB tensor.

    The actual CSV has far fewer than 1486 features.

    Therefore the normalized feature vector is tiled
    across the required width.

    Channel design:
        R = normalized feature values
        G = reversed feature values
        B = zero

    This keeps the input compatible with the existing
    9 x 1486 x 3 CNN.
    """

    values = np.asarray(
        feature_vector,
        dtype=np.float32,
    )

    # Normalize using dataset-wide feature bounds.
    denominator = (
        feature_max -
        feature_min
    )

    denominator[
        denominator < 1e-12
    ] = 1.0

    values = (
        values - feature_min
    ) / denominator

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    values = np.clip(
        values,
        0.0,
        1.0,
    )

    # Repeat feature vector until width is filled.
    repeated = np.resize(
        values,
        IMAGE_WIDTH,
    )

    # Create 9 rows with small deterministic
    # cyclic shifts. This avoids producing nine
    # identical rows while retaining the same flow
    # representation.
    rows = np.empty(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ),
        dtype=np.float32,
    )

    for row in range(IMAGE_HEIGHT):

        shift = (
            row * len(values)
        ) % IMAGE_WIDTH

        rows[row] = np.roll(
            repeated,
            shift,
        )

    # RGB representation.
    image = np.zeros(
        IMAGE_SHAPE,
        dtype=np.uint8,
    )

    image[:, :, 0] = (
        rows * 255.0
    ).astype(np.uint8)

    image[:, :, 1] = (
        np.flip(
            rows,
            axis=1,
        ) * 255.0
    ).astype(np.uint8)

    # Blue remains zero.
    image[:, :, 2] = 0

    return image


# ============================================================
# DATASET IMAGE CREATION
# ============================================================

def create_images(
    feature_df,
    labels,
    feature_min,
    feature_max,
):
    images = []

    for index in range(
        len(feature_df)
    ):

        if index % 5000 == 0:
            print(
                f"[*] Converting flows: "
                f"{index:,}/{len(feature_df):,}"
            )

        vector = feature_df.iloc[
            index
        ].to_numpy(
            dtype=np.float32
        )

        image = features_to_image(
            vector,
            feature_min,
            feature_max,
        )

        images.append(image)

    if not images:
        raise RuntimeError(
            "No images were generated."
        )

    X = np.stack(
        images
    ).astype(
        np.uint8
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    return X, y


# ============================================================
# BALANCED SAMPLING
# ============================================================

def balance_classes(
    X,
    y,
    max_images_per_class,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    selected_indices = []

    print()
    print("=" * 70)
    print("[*] CLASS BALANCING")
    print("=" * 70)

    for class_id in range(
        NUM_CLASSES
    ):

        indices = np.where(
            y == class_id
        )[0]

        if len(indices) == 0:
            print(
                f"[!] {CLASS_NAMES[class_id]:15s}: "
                f"ZERO samples"
            )
            continue

        rng.shuffle(indices)

        if max_images_per_class is None:
            selected = indices
        else:
            selected = indices[
                :max_images_per_class
            ]

        selected_indices.extend(
            selected.tolist()
        )

        print(
            f"    {CLASS_NAMES[class_id]:15s}: "
            f"{len(selected):,}"
        )

    if not selected_indices:
        raise RuntimeError(
            "No samples survived class balancing."
        )

    rng.shuffle(
        selected_indices
    )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    return (
        X[selected_indices],
        y[selected_indices],
    )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    X,
    y,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in range(
        NUM_CLASSES
    ):

        indices = np.where(
            y == class_id
        )[0]

        rng.shuffle(indices)

        n = len(indices)

        if n == 1:
            train_end = 1
            val_end = 1

        elif n == 2:
            train_end = 1
            val_end = 1

        else:
            train_end = max(
                1,
                int(
                    n * train_ratio
                ),
            )

            val_count = max(
                1,
                int(
                    n * val_ratio
                ),
            )

            val_end = min(
                n - 1,
                train_end + val_count,
            )

        train_indices.extend(
            indices[:train_end]
            .tolist()
        )

        val_indices.extend(
            indices[
                train_end:val_end
            ].tolist()
        )

        test_indices.extend(
            indices[val_end:]
            .tolist()
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        val_indices
    )

    rng.shuffle(
        test_indices
    )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    val_indices = np.asarray(
        val_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    return {
        "train": (
            X[train_indices],
            y[train_indices],
        ),
        "val": (
            X[val_indices],
            y[val_indices],
        ),
        "test": (
            X[test_indices],
            y[test_indices],
        ),
    }


# ============================================================
# PRINT SPLIT DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    y,
):
    print()
    print(
        f"[*] {name} distribution"
    )

    counts = Counter(
        y.tolist()
    )

    for class_id in range(
        NUM_CLASSES
    ):
        print(
            f"    {CLASS_NAMES[class_id]:15s}: "
            f"{counts.get(class_id, 0):,}"
        )


# ============================================================
# MAIN DATASET BUILDER
# ============================================================

def build_dataset(
    csv_paths,
    out_dir,
    max_images_per_class=2500,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    print()
    print("=" * 70)
    print(
        "CSV-ONLY 9-CLASS DATASET BUILDER"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD ALL CSV FILES
    # --------------------------------------------------------

    all_features = []
    all_labels = []

    for csv_path in csv_paths:

        feature_df, labels = (
            load_csv_file(
                csv_path
            )
        )

        all_features.append(
            feature_df
        )

        all_labels.append(
            labels
        )

    if not all_features:
        raise RuntimeError(
            "No CSV files were loaded."
        )

    # --------------------------------------------------------
    # ALIGN FEATURES
    # --------------------------------------------------------

    print()
    print(
        "[*] Combining CSV files..."
    )

    combined_features = pd.concat(
        all_features,
        ignore_index=True,
        sort=False,
    )

    combined_labels = np.concatenate(
        all_labels
    )

    # Convert any newly introduced missing
    # columns to zero.
    combined_features = (
        combined_features
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # Numeric conversion.
    combined_features = (
        combined_features
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .astype(np.float32)
    )

    print(
        f"[*] Combined flows: "
        f"{len(combined_labels):,}"
    )

    print(
        f"[*] Feature count: "
        f"{combined_features.shape[1]}"
    )

   # --------------------------------------------------------
# REMOVE CONSTANT / DUPLICATE COLUMNS
# --------------------------------------------------------

# Duplicate feature columns can exist in
# CICFlowMeter exports.
    # --------------------------------------------------------
# REMOVE DUPLICATE FEATURE COLUMNS
# --------------------------------------------------------

# CICFlowMeter exports can contain duplicate feature
# columns. Remove duplicate COLUMNS only.
#
# IMPORTANT:
# Do NOT call DataFrame.drop_duplicates() here.
# That attempts to compare millions of rows and can
# consume huge amounts of memory/time.

    before = combined_features.shape[1]

    combined_features = combined_features.loc[
        :,
        ~combined_features.columns.duplicated()
    ].copy()

    after = combined_features.shape[1]

    if after != before:
        print(
            f"[*] Removed "
            f"{before - after} duplicate "
            f"feature columns."
        )


    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURE COLUMNS
    # --------------------------------------------------------

    # Remove columns whose values never change.
    # This is much cheaper than row-level deduplication.

        constant_columns = [
            column
            for column in combined_features.columns
            if combined_features[column].nunique(
                dropna=False
            ) <= 1
        ]

        if constant_columns:
            combined_features = combined_features.drop(
                columns=constant_columns
            )

            print(
                f"[*] Removed "
                f"{len(constant_columns)} constant "
                f"feature columns."
            )


# --------------------------------------------------------
# REMOVE INVALID ROWS
# --------------------------------------------------------

    finite_mask = np.isfinite(
        combined_features.to_numpy(
            dtype=np.float64,
            copy=False
        )
    ).all(axis=1)

    combined_features = (
        combined_features.loc[
            finite_mask
        ]
        .reset_index(drop=True)
    )

    combined_labels = (
        combined_labels[
            finite_mask
        ]
    )

    print(
        f"[*] Valid rows after cleaning: "
        f"{len(combined_features):,}"
    )

    print(
        f"[*] Final feature count: "
        f"{combined_features.shape[1]}"
    )

    # --------------------------------------------------------
    # FEATURE NORMALIZATION
    # --------------------------------------------------------

    feature_array = (
        combined_features.to_numpy(
            dtype=np.float32
        )
    )

    feature_min = np.nanmin(
        feature_array,
        axis=0,
    )

    feature_max = np.nanmax(
        feature_array,
        axis=0,
    )

    feature_min = np.nan_to_num(
        feature_min,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    feature_max = np.nan_to_num(
        feature_max,
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # SAMPLE BEFORE IMAGE CREATION
    # --------------------------------------------------------

    # We cap BEFORE creating the huge image arrays.
    # This saves a large amount of RAM.
    selected_indices = []

    print()
    print(
        "=" * 70
    )
    print(
        "[*] SELECTING BALANCED FLOWS"
    )
    print(
        "=" * 70
    )

    for class_id in range(
        NUM_CLASSES
    ):

        indices = np.where(
            combined_labels == class_id
        )[0]

        rng.shuffle(indices)

        if len(indices) == 0:
            print(
                f"[!] {CLASS_NAMES[class_id]:15s}: "
                f"ZERO"
            )
            continue

        if max_images_per_class is None:
            selected = indices
        else:
            selected = indices[
                :max_images_per_class
            ]

        selected_indices.extend(
            selected.tolist()
        )

        print(
            f"    {CLASS_NAMES[class_id]:15s}: "
            f"{len(selected):,}"
        )

    if not selected_indices:
        raise RuntimeError(
            "No labelled flows available."
        )

    rng.shuffle(
        selected_indices
    )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    selected_features = (
        feature_array[
            selected_indices
        ]
    )

    selected_labels = (
        combined_labels[
            selected_indices
        ]
    )

    print(
        f"\n[*] Selected total flows: "
        f"{len(selected_labels):,}"
    )

    # --------------------------------------------------------
    # CREATE IMAGES
    # --------------------------------------------------------

    X, y = create_images(
        pd.DataFrame(
            selected_features
        ),
        selected_labels,
        feature_min,
        feature_max,
    )

    print()
    print(
        f"[*] Image dataset shape: "
        f"{X.shape}"
    )

    print(
        f"[*] Label shape: "
        f"{y.shape}"
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    splits = stratified_split(
        X,
        y,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=seed,
    )

    train_X, train_y = splits[
        "train"
    ]

    val_X, val_y = splits[
        "val"
    ]

    test_X, test_y = splits[
        "test"
    ]

    print_distribution(
        "TRAIN",
        train_y,
    )

    print_distribution(
        "VALIDATION",
        val_y,
    )

    print_distribution(
        "TEST",
        test_y,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    train_path = os.path.join(
        out_dir,
        "train.npz",
    )

    val_path = os.path.join(
        out_dir,
        "val.npz",
    )

    test_path = os.path.join(
        out_dir,
        "test.npz",
    )

    print()
    print(
        "[*] Saving dataset..."
    )

    np.savez_compressed(
        train_path,
        X=train_X,
        y=train_y,
    )

    np.savez_compressed(
        val_path,
        X=val_X,
        y=val_y,
    )

    np.savez_compressed(
        test_path,
        X=test_X,
        y=test_y,
    )

    # --------------------------------------------------------
    # SAVE FEATURE METADATA
    # --------------------------------------------------------

    metadata_path = os.path.join(
        out_dir,
        "feature_metadata.npz",
    )

    np.savez_compressed(
        metadata_path,
        feature_min=feature_min,
        feature_max=feature_max,
    )

    print()
    print("=" * 70)
    print(
        "[+] DATASET BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        f"[*] Train: {train_path}"
    )

    print(
        f"[*] Validation: {val_path}"
    )

    print(
        f"[*] Test: {test_path}"
    )

    print(
        f"[*] Metadata: {metadata_path}"
    )

    print(
        f"[*] Total images: {len(X):,}"
    )

    print(
        f"[*] Image shape: {X.shape[1:]}"
    )

    print("=" * 70)


# ============================================================
# SELF TEST
# ============================================================

def self_test():
    assert len(CLASS_NAMES) == 9

    assert map_label(
        "BENIGN"
    ) == 0

    assert map_label(
        "DoS Hulk"
    ) == 1

    assert map_label(
        "DDoS"
    ) == 2

    assert map_label(
        "PortScan"
    ) == 3

    assert map_label(
        "FTP-Patator"
    ) == 4

    assert map_label(
        "SSH-Patator"
    ) == 4

    assert map_label(
        "Bot"
    ) == 5

    assert map_label(
        "Web Attack - XSS"
    ) == 6

    assert map_label(
        "Web Attack - SQL Injection"
    ) == 6

    assert map_label(
        "Infiltration"
    ) == 7

    assert map_label(
        "Heartbleed"
    ) == 8

    # Small image test.
    vector = np.arange(
        78,
        dtype=np.float32,
    )

    feature_min = np.zeros(
        78,
        dtype=np.float32,
    )

    feature_max = np.ones(
        78,
        dtype=np.float32,
    ) * 100.0

    image = features_to_image(
        vector,
        feature_min,
        feature_max,
    )

    assert image.shape == IMAGE_SHAPE
    assert image.dtype == np.uint8

    print(
        "[+] Self-test passed."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build 9-class RGB image dataset "
            "directly from CIC-IDS2017 CSV files."
        )
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=False,
        help="CIC-IDS2017 labelled CSV files.",
    )

    parser.add_argument(
        "--out",
        default=os.path.join(
            "dataset",
            "images_final",
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=2500,
        help="Maximum samples per class.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test.",
    )

    args = parser.parse_args()

    if args.self_test or not args.labels:
        self_test()

    else:

        cap = (
            args.max_images_per_class
            if args.max_images_per_class > 0
            else None
        )

        build_dataset(
            csv_paths=args.labels,
            out_dir=args.out,
            max_images_per_class=cap,
            seed=args.seed,
        )