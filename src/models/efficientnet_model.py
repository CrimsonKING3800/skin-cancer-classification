"""
efficientnet_model.py - EfficientNet-B3 for skin lesion classification.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3

def build_efficientnet_model(input_shape=(128, 128, 3), num_classes=1):
    base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out, name="EfficientNetB3_SkinCancer")
