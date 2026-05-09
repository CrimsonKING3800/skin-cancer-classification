"""
resnet_model.py — ResNet50 model for skin lesion classification.

Uses transfer learning with ImageNet weights, fine-tuned on dermoscopic images.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


def build_resnet_model(input_shape=(128, 128, 3), num_classes=1, trainable_layers=20):
    """
    Build a ResNet50 model for binary skin lesion classification.

    Args:
        input_shape: Input image dimensions.
        num_classes: Number of output classes (1 for binary sigmoid).
        trainable_layers: Number of top layers to fine-tune.

    Returns:
        keras.Model
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze early layers, fine-tune the last N layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=output, name="ResNet50_SkinCancer")
