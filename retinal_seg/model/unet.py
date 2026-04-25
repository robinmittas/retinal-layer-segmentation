"""U-Net with residual blocks for multi-class retinal layer segmentation.

Architecture overview:
    6-level encoder  (16 → 512 filters, each level halves spatial dims)
    2-block bottleneck (1024 filters, optional non-local self-attention)
    6-level decoder  (512 → 16 filters, with skip connections from encoder)
    1×1 Conv output  (n_classes channels + softmax)

Reference:
    Ronneberger et al. (2015) U-Net: Convolutional Networks for
    Biomedical Image Segmentation. https://arxiv.org/abs/1505.04597
"""
from __future__ import annotations

from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, concatenate
from tensorflow.keras.models import Model

from retinal_seg.model.blocks import non_local_block, resconv
from retinal_seg.model.losses import dice_coef, weighted_ce_dice

K.set_image_data_format("channels_last")

_ENCODER_FILTERS: List[int] = [16, 32, 64, 128, 256, 512]
_BOTTLENECK_FILTERS: int = 1024
_NUM_BLOCKS: int = 14  # 6 encoder + 2 bottleneck + 6 decoder


def build_unet(
    img_height: int = 256,
    img_width: int = 256,
    img_channels: int = 1,
    n_classes: int = 1,
    last_activation: str = "softmax",
    pretrained_weights: Optional[str] = None,
    compile_model: bool = True,
    bn_list: Optional[List[bool]] = None,
    drop_rate: Union[float, List[float]] = 0.0,
    non_local_attention: bool = False,
    momentum: float = 0.8,
    loss: Callable = weighted_ce_dice,
) -> Model:
    """Build and optionally compile a 6-level residual U-Net.

    Args:
        img_height: Input image height in pixels.
        img_width: Input image width in pixels.
        img_channels: Number of input channels (1 for grayscale OCT).
        n_classes: Number of segmentation output classes.
        last_activation: Final layer activation; 'softmax' for multi-class,
            'sigmoid' for binary segmentation.
        pretrained_weights: Path to a .h5 weights checkpoint, or None.
        compile_model: If True, compile with Adam and dice_coef metric.
        bn_list: List of 14 booleans enabling BatchNorm per residual block
            (encoder blocks 1–6, bottleneck blocks 7–8, decoder blocks 9–14).
            Defaults to all-False when None.
        drop_rate: Scalar applied to all blocks, or a 14-element list
            specifying a per-block rate.
        non_local_attention: If True, insert a non-local self-attention block
            at the bottleneck between the two bottleneck resconv blocks.
        momentum: BatchNormalization momentum.
        loss: Loss function used during compilation.

    Returns:
        Compiled (or uncompiled) Keras Model.
    """
    bn = _expand_param(bn_list, default=False)
    dr = _expand_param(
        drop_rate if isinstance(drop_rate, list) else None,
        default=float(drop_rate) if drop_rate else 0.0,
    )

    inputs = Input((img_height, img_width, img_channels))

    # ------------------------------------------------------------------
    # Encoder path
    # ------------------------------------------------------------------
    e1 = resconv(inputs, _ENCODER_FILTERS[0], "enc1",
                 residual=False, batch_norm=bn[0], drop_rate=dr[0], momentum=momentum)
    e2 = resconv(MaxPooling2D((2, 2))(e1), _ENCODER_FILTERS[1], "enc2",
                 residual=False, batch_norm=bn[1], drop_rate=dr[1], momentum=momentum)
    e3 = resconv(MaxPooling2D((2, 2))(e2), _ENCODER_FILTERS[2], "enc3",
                 residual=False, batch_norm=bn[2], drop_rate=dr[2], momentum=momentum)
    e4 = resconv(MaxPooling2D((2, 2))(e3), _ENCODER_FILTERS[3], "enc4",
                 residual=False, batch_norm=bn[3], drop_rate=dr[3], momentum=momentum)
    e5 = resconv(MaxPooling2D((2, 2))(e4), _ENCODER_FILTERS[4], "enc5",
                 residual=False, batch_norm=bn[4], drop_rate=dr[4], momentum=momentum)
    e6 = resconv(MaxPooling2D((2, 2))(e5), _ENCODER_FILTERS[5], "enc6",
                 residual=False, batch_norm=bn[5], drop_rate=dr[5], momentum=momentum)

    # ------------------------------------------------------------------
    # Bottleneck
    # ------------------------------------------------------------------
    b = resconv(MaxPooling2D((2, 2))(e6), _BOTTLENECK_FILTERS, "btn_start",
                residual=False, batch_norm=bn[6], drop_rate=dr[6], momentum=momentum)

    if non_local_attention:
        b = non_local_block(b)

    b = resconv(b, _BOTTLENECK_FILTERS, "btn_end",
                residual=False, batch_norm=bn[7], drop_rate=dr[7], momentum=momentum)

    # ------------------------------------------------------------------
    # Decoder path (skip connections via concatenate)
    # output_padding=(1, 0) corrects for asymmetric spatial dims (512×496)
    # ------------------------------------------------------------------
    d6 = resconv(
        concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), output_padding=(1, 0))(b), e6]),
        512, "dec6", residual=False, batch_norm=bn[8], drop_rate=dr[8], momentum=momentum,
    )
    d7 = resconv(
        concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), output_padding=(1, 0))(d6), e5]),
        256, "dec7", residual=False, batch_norm=bn[9], drop_rate=dr[9], momentum=momentum,
    )
    d8 = resconv(
        concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(d7), e4]),
        128, "dec8", residual=False, batch_norm=bn[10], drop_rate=dr[10], momentum=momentum,
    )
    d9 = resconv(
        concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(d8), e3]),
        64, "dec9", residual=False, batch_norm=bn[11], drop_rate=dr[11], momentum=momentum,
    )
    d10 = resconv(
        concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(d9), e2]),
        32, "dec10", residual=False, batch_norm=bn[12], drop_rate=dr[12], momentum=momentum,
    )
    d11 = resconv(
        concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(d10), e1]),
        16, "dec11", residual=False, batch_norm=bn[13], drop_rate=dr[13], momentum=momentum,
    )

    outputs = Conv2D(n_classes, (1, 1), activation=last_activation, name="final_conv")(d11)

    model = Model(inputs=[inputs], outputs=[outputs])

    if compile_model:
        model.compile(optimizer="adam", loss=[loss], metrics=[dice_coef])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def _expand_param(
    param: Optional[List],
    default: Union[bool, float],
) -> List:
    """Return a _NUM_BLOCKS-length list from an explicit list or a scalar default.

    Args:
        param: Pre-built list of length _NUM_BLOCKS, or None.
        default: Scalar value used to fill all positions when param is None.

    Returns:
        List of length _NUM_BLOCKS.
    """
    if param is not None:
        return list(param)
    return [default] * _NUM_BLOCKS
