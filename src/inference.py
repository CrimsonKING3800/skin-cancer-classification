"""
inference.py — Single-image inference for skin cancer classification.

Usage:
    python src/inference.py --image /path/to/lesion.jpg --model outputs/best_model.keras
"""

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from preprocessing import preprocess_pipeline


def load_model(model_path: str):
    """Load a trained Keras model."""
    return tf.keras.models.load_model(model_path)


def predict_single_image(model, image_path: str, tabular_features: np.ndarray = None):
    """
    Run inference on a single dermoscopic image.

    Args:
        model: Trained Keras model.
        image_path: Path to the input image.
        tabular_features: Optional array of 34 tabular features.

    Returns:
        Probability of malignancy (float).
    """
    # Load and preprocess image
    img = np.array(Image.open(image_path).convert("RGB"))
    img = preprocess_pipeline(img, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    if tabular_features is not None:
        tabular_features = np.expand_dims(tabular_features, axis=0).astype(np.float32)
        prediction = model.predict({"image": img, "tab": tabular_features})
    else:
        prediction = model.predict(img)

    return float(prediction[0][0])


def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to dermoscopic image")
    parser.add_argument("--model", type=str, default="outputs/best_model.keras", help="Path to model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    print(f"Running inference on: {args.image}")
    probability = predict_single_image(model, args.image)

    label = "🔴 MALIGNANT" if probability >= args.threshold else "🟢 BENIGN"
    print(f"\n  Prediction: {label}")
    print(f"  Malignancy Probability: {probability:.4f}")
    print(f"  Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
