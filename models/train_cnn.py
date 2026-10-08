"""
train_cnn.py

9-class CNN training for the Real-Time Network Intrusion Detection System.

Classes:
    0 - Normal
    1 - DoS
    2 - DDoS
    3 - Port Scan
    4 - Brute Force
    5 - Botnet
    6 - Web Attack
    7 - Infiltration
    8 - Heartbleed

Expected dataset:
    dataset/images/train.npz
    dataset/images/val.npz
    dataset/images/test.npz

Each file must contain:
    X -> (N, 9, 1486, 3) uint8
    y -> (N,) integer labels

Usage:
    python models/train_cnn.py
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

IMG_HEIGHT = 9
IMG_WIDTH = 1486
IMG_CHANNELS = 3

NUM_CLASSES = 9

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

DATASET_DIR = os.path.join("dataset", "images_final")

MODEL_OUT_DIR = "models"

KERAS_MODEL_PATH = os.path.join(
    MODEL_OUT_DIR,
    "cnn_model.keras"
)


# ============================================================
# BUILD CNN
# ============================================================

def build_model() -> tf.keras.Model:
    """
    Lightweight 9-class CNN.

    Input:
        9 sequential packets
        x 1486 bytes/features
        x 3 RGB channels

    Output:
        9-class softmax probability vector.
    """

    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            IMG_CHANNELS
        )
    )

    # --------------------------------------------------------
    # Normalize uint8 [0,255] -> float [0,1]
    # --------------------------------------------------------

    x = layers.Rescaling(
        1.0 / 255.0
    )(inputs)

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    x = layers.Dropout(
        0.2
    )(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    x = layers.Dropout(
        0.2
    )(x)

    # --------------------------------------------------------
    # Global feature extraction
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    # Fully connected layer
    # --------------------------------------------------------

    x = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="glorot_normal"
    )(x)

    # --------------------------------------------------------
    # 9-class output
    # --------------------------------------------------------

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="real_time_nids_cnn_9class"
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# LOAD DATASET
# ============================================================

def load_split(name: str):
    """
    Load train / validation / test dataset.
    """

    path = os.path.join(
        DATASET_DIR,
        f"{name}.npz"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[!] Dataset file not found:\n"
            f"    {path}\n\n"
            f"Run data_prep/pcap_to_dataset.py first."
        )

    print(f"[*] Loading {name} dataset...")

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    print(f"[+] {name} X shape: {X.shape}")
    print(f"[+] {name} y shape: {y.shape}")

    return X, y


# ============================================================
# DATASET CHECK
# ============================================================

def print_class_distribution(y, name):
    """
    Display the number of samples belonging to each class.
    """

    print(f"\n[*] {name} class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = int(
            np.sum(y == class_id)
        )

        print(
            f"    {class_id}: "
            f"{class_name:<15} "
            f"{count}"
        )


# ============================================================
# TRAIN
# ============================================================

def train():

    print("\n" + "=" * 60)
    print(" REAL-TIME NIDS - 9 CLASS CNN TRAINING")
    print("=" * 60)

    print("\n[*] Classes:")

    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i} -> {name}")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    X_train, y_train = load_split("train")

    X_val, y_val = load_split("val")

    # --------------------------------------------------------
    # Print distributions
    # --------------------------------------------------------

    print_class_distribution(
        y_train,
        "Training"
    )

    print_class_distribution(
        y_val,
        "Validation"
    )

    # --------------------------------------------------------
    # Validate image shape
    # --------------------------------------------------------

    expected_shape = (
        IMG_HEIGHT,
        IMG_WIDTH,
        IMG_CHANNELS
    )

    if X_train.ndim != 4:
        raise ValueError(
            f"Expected training images to have 4 dimensions, "
            f"got {X_train.ndim}"
        )

    if tuple(X_train.shape[1:]) != expected_shape:
        raise ValueError(
            f"Unexpected training image shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got: {X_train.shape[1:]}"
        )

    if tuple(X_val.shape[1:]) != expected_shape:
        raise ValueError(
            f"Unexpected validation image shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got: {X_val.shape[1:]}"
        )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    unique_train = np.unique(y_train)

    unique_val = np.unique(y_val)

    print(
        f"\n[*] Training labels present: "
        f"{unique_train.tolist()}"
    )

    print(
        f"[*] Validation labels present: "
        f"{unique_val.tolist()}"
    )

    missing_train = [
        i
        for i in range(NUM_CLASSES)
        if i not in unique_train
    ]

    if missing_train:

        print(
            "\n[!] WARNING:"
            "\n    The training dataset is missing:"
        )

        for i in missing_train:
            print(
                f"    {i} -> {CLASS_NAMES[i]}"
            )

        print(
            "\n[!] The CNN can still train, but "
            "the missing classes cannot be learned."
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n[*] Building 9-class CNN...")

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_OUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            KERAS_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" STARTING TRAINING")
    print("=" * 60)

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=15,

        batch_size=64,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        KERAS_MODEL_PATH
    )

    print(
        f"\n[+] Saved trained model to:"
        f"\n    {KERAS_MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    try:

        X_test, y_test = load_split("test")

        print_class_distribution(
            y_test,
            "Test"
        )

        print(
            "\n[*] Evaluating on test dataset..."
        )

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=1
        )

        print(
            f"\n[+] Test accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"[+] Test loss: "
            f"{loss:.4f}"
        )

    except FileNotFoundError as e:

        print(
            f"\n[!] Test evaluation skipped:"
            f"\n{e}"
        )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nModel:"
        f"\n    {KERAS_MODEL_PATH}"
    )

    print("\nClasses:")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"    {i} -> {name}"
        )

    print()


# ============================================================
# SELF TEST
# ============================================================

def _self_test():

    """
    Verify that the 9-class CNN can build, compile,
    train on dummy data, and produce 9-class probabilities.
    """

    print(
        "\n[*] Running 9-class CNN self-test..."
    )

    model = build_model()

    n = 16

    X_dummy = np.random.randint(
        0,
        256,
        size=(
            n,
            IMG_HEIGHT,
            IMG_WIDTH,
            IMG_CHANNELS
        ),
        dtype=np.uint8
    )

    y_dummy = np.random.randint(
        0,
        NUM_CLASSES,
        size=(n,)
    )

    history = model.fit(
        X_dummy,
        y_dummy,
        epochs=1,
        batch_size=4,
        verbose=0
    )

    assert "loss" in history.history

    output = model.predict(
        X_dummy[:2],
        verbose=0
    )

    # Must produce 2 samples x 9 classes
    assert output.shape == (
        2,
        NUM_CLASSES
    )

    # Softmax probabilities must sum to 1
    assert np.allclose(
        output.sum(axis=1),
        1.0,
        atol=1e-3
    )

    print(
        f"[+] Model parameters: "
        f"{model.count_params():,}"
    )

    print(
        f"[+] Output shape: "
        f"{output.shape}"
    )

    print(
        "[+] 9-class CNN self-test passed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if os.path.exists(DATASET_DIR):

        train()

    else:

        print(
            f"[!] {DATASET_DIR} does not exist."
        )

        print(
            "[*] Running architecture self-test instead."
        )

        _self_test()