"""
train.py — Training script for skin cancer classification models.

Usage:
    python src/train.py --data_path /path/to/data --epochs 20 --batch_size 32
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from sklearn.preprocessing import StandardScaler

from models.ensemble import build_vit_tabular_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin Cancer Classifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory for saved models")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_callbacks(output_dir: str):
    """Create training callbacks."""
    return [
        ModelCheckpoint(
            filepath=f"{output_dir}/best_model.keras",
            monitor="val_recall",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_recall",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def main():
    args = parse_args()
    set_seed(args.seed)

    # Enable mixed precision on GPU
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("✓ Using mixed precision (float16)")

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # ── Data Loading ─────────────────────────────────────────────
    # NOTE: Replace this section with your actual data loading logic.
    # The notebooks load from HDF5 files; adapt paths as needed.
    print(f"Loading data from: {args.data_path}")
    print("⚠️  Update data loading in train.py to match your file paths.")

    # ── Model ────────────────────────────────────────────────────
    model = build_vit_tabular_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    print("\n✓ Model compiled. Ready for training.")
    print("  To train, provide preprocessed data arrays to model.fit().")


if __name__ == "__main__":
    main()
