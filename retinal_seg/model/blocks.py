"""Reusable Keras building blocks: residual convolution and non-local attention.

References:
    Wang et al. (2018) Non-local Means Neural Networks
    https://arxiv.org/abs/1711.07971
"""
from __future__ import annotations

import keras
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    Dropout,
    MaxPool1D,
    Reshape,
    dot,
)

_KERNEL_INIT = "he_normal"


def resconv(
    input_layer: tf.Tensor,
    out_dim: int,
    name: str,
    residual: bool = True,
    drop_rate: float = 0.0,
    batch_norm: bool = True,
    drop_trainable: bool = True,
    momentum: float = 0.8,
) -> tf.Tensor:
    """Residual convolution block with two 3×3 ELU-activated Conv2D layers.

    Applies two convolutions, optional BatchNormalization, optional
    MC-Dropout, and an optional residual (skip) connection back to the
    input.

    Note:
        The residual connection requires input_layer and the output to share
        the same channel count. Set residual=False when changing feature depth
        (e.g. every encoder/decoder transition).

    Args:
        input_layer: Input tensor of dtype float32, shape (batch, H, W, in_channels).
        out_dim: Number of output feature maps for both Conv2D layers.
        name: Base name prefix; layers are named <name>_1 and <name>_2.
        residual: If True, adds input_layer to the output (skip connection).
        drop_rate: Dropout rate applied after BatchNorm; 0.0 disables it.
        batch_norm: If True, applies BatchNormalization after the Conv layers.
        drop_trainable: If True, Dropout is active at inference (MC-Dropout).
        momentum: BatchNormalization momentum.

    Returns:
        Tensor of dtype float32, shape (batch, H, W, out_dim), or same shape
        as input_layer when residual=True.
    """
    x = Conv2D(
        out_dim, 3, activation="elu", padding="same",
        kernel_initializer=_KERNEL_INIT, name=f"{name}_1",
    )(input_layer)
    x = Conv2D(
        out_dim, 3, activation="elu", padding="same",
        kernel_initializer=_KERNEL_INIT, name=f"{name}_2",
    )(x)

    if batch_norm:
        x = BatchNormalization(
            axis=-1, center=False, scale=False, momentum=momentum
        )(x)

    if drop_rate:
        x = Dropout(drop_rate, trainable=drop_trainable)(x, training=True)

    if residual:
        x = Add()([input_layer, x])

    return x


def non_local_block(
    ip: tf.Tensor,
    intermediate_dim: int | None = None,
    compression: int = 2,
    mode: str = "embedded",
    add_residual: bool = True,
) -> tf.Tensor:
    """Non-local self-attention block for long-range feature dependency modelling.

    Computes pairwise relationships between all spatial positions in the
    feature map. Supports rank-3 (temporal), rank-4 (spatial), and
    rank-5 (spatio-temporal) inputs.

    Args:
        ip: Input tensor of dtype float32; rank 3 (batch, L, C),
            rank 4 (batch, H, W, C), or rank 5 (batch, D, H, W, C).
        intermediate_dim: Channels in the intermediate projection.
            Defaults to channels // 2 (minimum 1).
        compression: Spatial compression factor applied to the key/value
            paths to reduce memory. 1 disables compression.
        mode: Attention variant - one of 'embedded', 'gaussian', or 'dot'.
        add_residual: If True, adds the original input to the output.

    Returns:
        Tensor of dtype float32 with the same shape as ip.

    Raises:
        ValueError: If mode is not recognised or input rank is not 3-5.
        NotImplementedError: If mode='concatenate'.
    """
    valid_modes = {"gaussian", "embedded", "dot", "concatenate"}
    if mode not in valid_modes:
        raise ValueError(f"`mode` must be one of {valid_modes}, got '{mode}'.")

    channel_dim = 1 if K.image_data_format() == "channels_first" else -1
    ip_shape = K.int_shape(ip)
    rank = len(ip_shape)

    if rank == 3:
        _, dim1, channels = ip_shape
        dim2 = dim3 = None
    elif rank == 4:
        if channel_dim == 1:
            _, channels, dim1, dim2 = ip_shape
        else:
            _, dim1, dim2, channels = ip_shape
        dim3 = None
    elif rank == 5:
        if channel_dim == 1:
            _, channels, dim1, dim2, dim3 = ip_shape
        else:
            _, dim1, dim2, dim3, channels = ip_shape
    else:
        raise ValueError(
            "Input rank must be 3 (temporal), 4 (spatial), or 5 (spatio-temporal)."
        )

    if intermediate_dim is None:
        intermediate_dim = max(1, channels // 2)
    else:
        intermediate_dim = max(1, int(intermediate_dim))

    compression = compression or 1

    # ------------------------------------------------------------------
    # Attention weight computation
    # ------------------------------------------------------------------
    if mode == "gaussian":
        x1 = Reshape((-1, channels))(ip)
        x2 = Reshape((-1, channels))(ip)
        f = Activation("softmax")(dot([x1, x2], axes=2))

    elif mode == "dot":
        theta = Reshape((-1, intermediate_dim))(_conv_nd(ip, rank, intermediate_dim))
        phi = Reshape((-1, intermediate_dim))(_conv_nd(ip, rank, intermediate_dim))
        f = dot([theta, phi], axes=2)
        size = K.int_shape(f)
        f = keras.layers.Lambda(lambda z: z / float(size[-1]))(f)

    elif mode == "concatenate":
        raise NotImplementedError("Concatenate mode is not yet implemented.")

    else:  # embedded gaussian (default)
        theta = Reshape((-1, intermediate_dim))(_conv_nd(ip, rank, intermediate_dim))
        phi = Reshape((-1, intermediate_dim))(_conv_nd(ip, rank, intermediate_dim))
        if compression > 1:
            phi = MaxPool1D(compression)(phi)
        f = Activation("softmax")(dot([theta, phi], axes=2))

    # ------------------------------------------------------------------
    # Value path and output projection
    # ------------------------------------------------------------------
    g = Reshape((-1, intermediate_dim))(_conv_nd(ip, rank, intermediate_dim))
    if compression > 1 and mode == "embedded":
        g = MaxPool1D(compression)(g)

    y = dot([f, g], axes=[2, 1])

    # Reshape output back to the spatial format of ip
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        shape = (
            (intermediate_dim, dim1, dim2)
            if channel_dim == 1
            else (dim1, dim2, intermediate_dim)
        )
        y = Reshape(shape)(y)
    else:
        shape = (
            (intermediate_dim, dim1, dim2, dim3)
            if channel_dim == 1
            else (dim1, dim2, dim3, intermediate_dim)
        )
        y = Reshape(shape)(y)

    y = _conv_nd(y, rank, channels)

    if add_residual:
        y = Add()([ip, y])

    return y


def _conv_nd(ip: tf.Tensor, rank: int, channels: int) -> tf.Tensor:
    """Apply a pointwise (1×…×1) convolution matching the spatial rank.

    Args:
        ip: Input tensor of dtype float32.
        rank: Spatial rank of ip (3, 4, or 5).
        channels: Number of output channels.

    Returns:
        Tensor of dtype float32 with `channels` feature maps.
    """
    kwargs = dict(padding="same", use_bias=False, kernel_initializer=_KERNEL_INIT)
    if rank == 3:
        return Conv1D(channels, 1, **kwargs)(ip)
    if rank == 4:
        return Conv2D(channels, (1, 1), **kwargs)(ip)
    return Conv3D(channels, (1, 1, 1), **kwargs)(ip)
