"""
train_cnn.py

Defines and trains the CNN that looks at the RGB images from
image_builder.py and decides: Normal / Port Scan / Brute Force / DDoS.

Architecture follows the base paper's Fig. 6 / Table VII (their best
hyperparameters after tuning), with one deliberate change: the paper
does BINARY classification (benign vs malicious, sigmoid output).
Your PPT's proposed system names 4 specific classes, so this version
ends in a 4-way softmax instead — worth a sentence in your report
("we extend the base paper's binary classifier to multi-class").

Expects a dataset already built by data_prep/pcap_to_dataset.py:
  dataset/images/train.npz  -> arrays 'X' (N,9,1486,3) uint8, 'y' (N,) int
  dataset/images/val.npz
  dataset/images/test.npz

Usage (once the real .npz files exist):
    python models/train_cnn.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT = 9          # P — sequential packets per image (image_builder.P_PACKETS)
IMG_WIDTH = 1486         # Q — bytes per packet (image_builder.Q_FEATURES)
IMG_CHANNELS = 3          # R, G, B
NUM_CLASSES = 4          # Normal, Port Scan, Brute Force, DDoS
CLASS_NAMES = ["Normal", "Port Scan", "Brute Force", "DDoS"]

DATASET_DIR = os.path.join("dataset", "images")
MODEL_OUT_DIR = "models"
KERAS_MODEL_PATH = os.path.join(MODEL_OUT_DIR, "cnn_model.keras")


def build_model() -> tf.keras.Model:
    """
    Table VII's best values:
      conv1: 96 filters,  (5,5) kernel
      conv2: 128 filters, (5,5) kernel
      conv3: 192 filters, (3,3) kernel
      conv4: 384 filters, (4,4) kernel
      dense: 64 units
      activation: ReLU (GlorotNormal init)
      batchnorm after the first pooling layer
      2 dropout layers at 0.2
    All conv layers use 'same' padding, stride (1,1), exactly as the
    paper specifies (Section IV-C) — this keeps spatial dims stable
    through the conv stack so pooling is what shrinks the image.
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = layers.Rescaling(1.0 / 255.0)(inputs)  # uint8 0-255 -> float 0-1

    x = layers.Conv2D(96, (5, 5), padding="same", activation="relu",
                       kernel_initializer="glorot_normal")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu",
                       kernel_initializer="glorot_normal")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(192, (3, 3), padding="same", activation="relu",
                       kernel_initializer="glorot_normal")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = layers.Conv2D(384, (4, 4), padding="same", activation="relu",
                       kernel_initializer="glorot_normal")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu",
                      kernel_initializer="glorot_normal")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="spin_ids_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",  # y is an int label, not one-hot
        metrics=["accuracy"],
    )
    return model


def load_split(name: str):
    path = os.path.join(DATASET_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data_prep/pcap_to_dataset.py first "
            f"to build it from your downloaded pcap."
        )
    data = np.load(path)
    return data["X"], data["y"]


def train():
    model = build_model()
    model.summary()

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            KERAS_MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,            # paper used 100 epochs; EarlyStopping will cut this short
        batch_size=256,        # paper's best batch size (Table VII)
        callbacks=callbacks,
    )

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model.save(KERAS_MODEL_PATH)
    print(f"[+] Saved trained model to {KERAS_MODEL_PATH}")

    # Quick test-set check so you see a number immediately, before
    # bothering with quantize.py's more detailed evaluation.
    try:
        X_test, y_test = load_split("test")
        loss, acc = model.evaluate(X_test, y_test)
        print(f"[+] Test accuracy: {acc:.4f}  (loss: {loss:.4f})")
    except FileNotFoundError as e:
        print(f"[!] Skipping test-set evaluation: {e}")


def _self_test():
    """
    Proves the architecture builds, compiles, and can fit on dummy data
    shaped exactly like real images will be — no real dataset needed.
    """
    model = build_model()
    n = 16
    X_dummy = np.random.randint(0, 256, size=(n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    y_dummy = np.random.randint(0, NUM_CLASSES, size=(n,))
    history = model.fit(X_dummy, y_dummy, epochs=1, batch_size=4, verbose=0)
    assert "loss" in history.history
    out = model.predict(X_dummy[:2], verbose=0)
    assert out.shape == (2, NUM_CLASSES)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-3)  # softmax sums to 1
    print(f"[self-test] model built OK. Params: {model.count_params():,}")
    print("train_cnn.py self-test passed.")


if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        train()
    else:
        print(f"[!] {DATASET_DIR} doesn't exist yet — running architecture "
              f"self-test on dummy data instead of real training.\n")
        _self_test()
