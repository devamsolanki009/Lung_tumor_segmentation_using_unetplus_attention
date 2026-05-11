"""Attention U-Net++ architecture for lung tumor segmentation.

Architecture overview
---------------------
- Encoder path: 4 downsampling blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool)
- Bottleneck: deepest convolutional block with Dropout
- Decoder path: 4 upsampling blocks with attention gates on skip connections
- Output: 1-channel sigmoid activation (binary segmentation)

Reference
---------
Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018.
"""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

from lung_tumor_segmentation.models.losses import (
    bce_dice_loss,
    dice_coefficient,
    iou_metric,
    precision_metric,
    recall_metric,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    name: str = "conv_block",
) -> tf.Tensor:
    """Two Conv2D → BatchNorm → ReLU layers."""
    x = layers.Conv2D(filters, kernel_size, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_act1")(x)

    x = layers.Conv2D(filters, kernel_size, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_act2")(x)
    return x


def _encoder_block(
    x: tf.Tensor,
    filters: int,
    name: str = "enc",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Conv block followed by MaxPool; returns (skip, downsampled)."""
    skip = _conv_block(x, filters, name=name)
    down = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(skip)
    return skip, down


def _attention_gate(
    x: tf.Tensor,
    gating: tf.Tensor,
    filters: int,
    name: str = "att",
) -> tf.Tensor:
    """Additive soft-attention gate.

    Parameters
    ----------
    x : tf.Tensor
        Skip connection feature map (from encoder).
    gating : tf.Tensor
        Gating signal (from decoder / upsampled path).
    filters : int
        Number of intermediate attention filters.
    """
    theta_x = layers.Conv2D(filters, kernel_size=1, padding="same", name=f"{name}_theta")(x)
    phi_g = layers.Conv2D(filters, kernel_size=1, padding="same", name=f"{name}_phi")(gating)

    # Upsample gating signal to match skip connection spatial dims
    phi_g = layers.UpSampling2D(size=(2, 2), name=f"{name}_upsample")(phi_g)

    add = layers.Add(name=f"{name}_add")([theta_x, phi_g])
    add = layers.Activation("relu", name=f"{name}_relu")(add)

    psi = layers.Conv2D(1, kernel_size=1, padding="same", name=f"{name}_psi")(add)
    psi = layers.Activation("sigmoid", name=f"{name}_sigmoid")(psi)

    return layers.Multiply(name=f"{name}_out")([x, psi])


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    use_attention: bool = True,
    name: str = "dec",
) -> tf.Tensor:
    """Upsample → (attention gate on skip) → concatenate → conv block."""
    x = layers.Conv2DTranspose(
        filters, kernel_size=2, strides=2, padding="same", name=f"{name}_up"
    )(x)

    if use_attention:
        skip = _attention_gate(skip, x, filters // 2, name=f"{name}_att")

    x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    x = _conv_block(x, filters, name=f"{name}_conv")
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    filters: List[int] = (64, 128, 256, 512),
    dropout_rate: float = 0.3,
    use_attention: bool = True,
) -> Model:
    """Build and return an Attention U-Net model (Functional API).

    Parameters
    ----------
    input_shape : tuple
        ``(H, W, C)`` — spatial dimensions + channels (typically 1 for grayscale).
    filters : list of int
        Number of filters at each encoder/decoder level.
    dropout_rate : float
        Dropout applied at the bottleneck.
    use_attention : bool
        Whether to use attention gates on skip connections.

    Returns
    -------
    tf.keras.Model
        Un-compiled model.
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # ---- Encoder -------------------------------------------------------
    skips = []
    x = inputs
    for i, f in enumerate(filters):
        skip, x = _encoder_block(x, f, name=f"enc{i+1}")
        skips.append(skip)

    # ---- Bottleneck ----------------------------------------------------
    x = _conv_block(x, filters[-1] * 2, name="bottleneck")
    x = layers.Dropout(rate=dropout_rate, name="bottleneck_drop")(x)

    # ---- Decoder -------------------------------------------------------
    for i, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        x = _decoder_block(x, skip, f, use_attention=use_attention, name=f"dec{i+1}")

    # ---- Output --------------------------------------------------------
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="AttentionUNet")
    return model


def get_model(config: dict) -> Model:
    """Build and compile the Attention U-Net from a config dict.

    Parameters
    ----------
    config : dict
        Full project config (loaded via load_config()).

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.
    """
    mcfg = config["model"]
    tcfg = config["training"]

    model = build_model(
        input_shape=tuple(mcfg["input_shape"]),
        filters=mcfg["filters"],
        dropout_rate=mcfg["dropout_rate"],
        use_attention=mcfg["use_attention"],
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=tcfg["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[
            dice_coefficient,
            iou_metric,
            precision_metric,
            recall_metric,
        ],
    )

    model.summary(line_length=100)
    return model
