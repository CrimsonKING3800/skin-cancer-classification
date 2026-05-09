"""
ensemble.py — ViT + Tabular fusion model for skin cancer classification.

Combines a Vision Transformer (image branch) with an MLP (tabular branch)
for multi-modal classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Hyperparameters ──────────────────────────────────────────────
IMG_SIZE = (224, 224, 3)
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)  # 196
EMBED_DIM = 256
NUM_HEADS = 8
MLP_DIM = 512
ENCODER_DEPTH = 8
DROP_RATE = 0.1
TAB_DIM = 34


# ── ViT Components ──────────────────────────────────────────────
class PatchExtractor(layers.Layer):
    """Extract non-overlapping patches using a Conv2D projection."""

    def __init__(self, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
        return x


class AddClassTokenAndPos(layers.Layer):
    """Prepend a learnable [CLS] token and add positional embeddings."""

    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.cls = self.add_weight(name="cls", shape=(1, 1, embed_dim), initializer="zeros", trainable=True)
        self.pos = self.add_weight(name="pos", shape=(1, num_patches + 1, embed_dim), initializer="random_normal", trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls, [batch_size, 1, tf.shape(x)[-1]])
        x = tf.concat([cls_tokens, x], axis=1)
        return x + self.pos


def transformer_encoder(x, num_heads=NUM_HEADS, mlp_dim=MLP_DIM, drop=DROP_RATE):
    """Single Transformer encoder block with PreNorm."""
    # Multi-Head Self-Attention
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(y, y)
    y = layers.Dropout(drop)(y)
    x = layers.Add()([x, y])

    # Feed-Forward Network
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(mlp_dim, activation="gelu")(y)
    y = layers.Dropout(drop)(y)
    y = layers.Dense(x.shape[-1])(y)
    y = layers.Dropout(drop)(y)
    return layers.Add()([x, y])


# ── Model Builder ────────────────────────────────────────────────
def build_vit_tabular_model(
    img_size=IMG_SIZE,
    tab_dim=TAB_DIM,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    encoder_depth=ENCODER_DEPTH,
    drop_rate=DROP_RATE,
):
    """
    Build the ViT + Tabular fusion model.

    Architecture:
        Image  → PatchExtract → Transformer × N → [CLS] token → Dropout
        Tabular → Dense → BN → Dropout → Dense → BN → Dropout
        [Image features || Tabular features] → Dense → BN → Dropout → Sigmoid

    Returns:
        keras.Model with inputs {"image", "tab"} and binary output.
    """
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    # ── Image Branch (ViT) ──
    img_in = layers.Input(shape=img_size, name="image")
    x = PatchExtractor(patch_size, embed_dim)(img_in)
    x = AddClassTokenAndPos(num_patches, embed_dim)(x)

    for _ in range(encoder_depth):
        x = transformer_encoder(x, num_heads, mlp_dim, drop_rate)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_token = x[:, 0]  # Extract [CLS] token
    img_feat = layers.Dropout(drop_rate, name="image_features")(cls_token)

    # ── Tabular Branch (MLP) ──
    tab_in = layers.Input(shape=(tab_dim,), name="tab")
    t = layers.Dense(128, activation="relu")(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.2)(t)
    t = layers.Dense(128, activation="relu")(t)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.2)(t)

    # ── Fusion ──
    fused = layers.Concatenate(name="fusion")([img_feat, t])
    z = layers.Dense(256, activation="relu")(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)

    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="pred")(z)

    return keras.Model(inputs={"image": img_in, "tab": tab_in}, outputs=out, name="ViT_Tab_Fusion")
