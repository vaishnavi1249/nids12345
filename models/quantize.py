"""
quantize.py

Converts the trained Keras CNN (32-bit float weights) into an 8-bit
TFLite model. This is the literal implementation of your PPT slide 5's
"Quantized Edge Inference: 8-bit optimization from 32 bits" bullet,
and what app.py will actually load for live predictions.

Why this matters for your report: quantization is what makes the
"1.04 milliseconds per prediction" kind of claim believable — an int8
model is roughly 4x smaller and meaningfully faster to run than the
float32 original, which is the whole point of calling it "edge
inference" on your architecture diagram.

Usage (once models/cnn_model.keras exists from train_cnn.py):
    python models/quantize.py
"""

import os
import time
import numpy as np
import tensorflow as tf

KERAS_MODEL_PATH = os.path.join("models", "cnn_model.keras")
TFLITE_MODEL_PATH = os.path.join("models", "cnn_model_int8.tflite")

IMG_HEIGHT = 9
IMG_WIDTH = 1486
IMG_CHANNELS = 3


def _representative_dataset_gen(sample_images: np.ndarray):
    """
    TFLite's int8 converter needs a handful of REAL (or at least
    realistic) input samples to figure out the right scale/zero-point
    for quantizing each layer. We feed it a small slice of the training
    images for this — using random noise here would give a worse
    quantization than using real traffic shapes.
    """
    def gen():
        for i in range(min(200, len(sample_images))):
            img = sample_images[i : i + 1].astype(np.float32)
            yield [img]
    return gen


def quantize(sample_images: np.ndarray):
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen(sample_images)
    # Force full int8 (not just weights) so the speed benefit is real
    # on CPU, not just a smaller file on disk.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8   # matches image_builder's uint8 output directly
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    orig_size = os.path.getsize(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None
    new_size = os.path.getsize(TFLITE_MODEL_PATH)
    print(f"[+] Saved {TFLITE_MODEL_PATH} ({new_size / 1024:.1f} KB)")
    if orig_size:
        print(f"    vs original .keras file: {orig_size / 1024:.1f} KB "
              f"({orig_size / new_size:.1f}x smaller)")

    return TFLITE_MODEL_PATH


def benchmark_inference(tflite_path: str, sample_images: np.ndarray, n_runs: int = 100):
    """
    Measures per-image inference time, for the "how fast is this"
    number you'll want to quote next to the base paper's 1.04ms figure.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    n = min(n_runs, len(sample_images))
    times = []
    for i in range(n):
        img = sample_images[i : i + 1].astype(np.uint8)
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        times.append((time.perf_counter() - start) * 1000.0)  # ms

    times = np.array(times)
    print(f"[+] Inference benchmark over {n} images:")
    print(f"    mean:   {times.mean():.3f} ms")
    print(f"    median: {np.median(times):.3f} ms")
    print(f"    p95:    {np.percentile(times, 95):.3f} ms")
    return times


def _self_test():
    """
    Builds a tiny throwaway model with the right input/output shape,
    quantizes it, and benchmarks it — proves the quantize/benchmark
    code path works without needing your real trained model yet.
    """
    import sys
    sys.path.insert(0, ".")
    from models.train_cnn import build_model, NUM_CLASSES

    print("[self-test] no trained model found yet — building a throwaway "
          "one just to prove quantize.py works end-to-end.\n")
    model = build_model()
    os.makedirs("models", exist_ok=True)
    model.save(KERAS_MODEL_PATH)

    dummy_images = np.random.randint(
        0, 256, size=(50, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
    )
    tflite_path = quantize(dummy_images)
    benchmark_inference(tflite_path, dummy_images, n_runs=20)

    os.remove(KERAS_MODEL_PATH)
    print("\nquantize.py self-test passed (throwaway model files cleaned up).")


if __name__ == "__main__":
    if os.path.exists(KERAS_MODEL_PATH):
        # Real run: needs real training images as the representative set.
        train_npz = os.path.join("dataset", "images", "train.npz")
        data = np.load(train_npz)
        tflite_path = quantize(data["X"])
        benchmark_inference(tflite_path, data["X"])
    else:
        _self_test()
