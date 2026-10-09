"""Bounded image decoding with the released model's 832x480 resize."""

import warnings
from io import BytesIO

import numpy as np
from PIL import Image


def fit_image(data: bytes) -> np.ndarray:
    """Decode once and resize with native PIL bilinear interpolation."""
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Upload a non-empty image smaller than 25 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP", "BMP"):
                    raise ValueError("Upload PNG, JPEG, WebP or BMP")
                if image.width * image.height > 40_000_000:
                    raise ValueError("Image must have at most 40 million pixels")
                return np.asarray(
                    image.convert("RGB").resize((832, 480), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                ).copy()
    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(
            "Upload a valid image with at most 40 million pixels"
        ) from error
