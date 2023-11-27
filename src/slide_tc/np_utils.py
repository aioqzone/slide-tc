import typing as t

import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image as image

from ..pil_utils import frombytes as pil_frombytes
from ..pil_utils import tobytes as pil_tobytes

if t.TYPE_CHECKING:
    mat_u1 = np.ndarray[t.Any, np.dtype[np.uint8]]
    mat_i2 = np.ndarray[t.Any, np.dtype[np.int16]]
    mat_i4 = np.ndarray[t.Any, np.dtype[np.int32]]
else:
    mat_u1 = mat_i2 = mat_i4 = np.ndarray

__all__ = [
    "mat_u1",
    "mat_i2",
    "mat_i4",
    "frombytes",
    "tobytes",
    "conv2d",
    "corr_norm_mask_match",
    "hdiff_corr_norm_mask_match",
    "vdiff_corr_norm_mask_match",
    "energy_mask",
    "sdiff_mask_match",
]


def frombytes(b: bytes, dtype=np.uint8) -> mat_u1:
    return np.asarray(pil_frombytes(b), dtype=dtype)


def tobytes(img: mat_u1, format="png") -> bytes:
    return pil_tobytes(image.fromarray(img), format=format)


def conv2d(x: np.ndarray, k: np.ndarray, axes=(0, 1)):
    """
    :return: f8 array, with the shape of `x.shape - (k.shape - 1)`
    """
    kh, kw = k.shape[axes[0]], k.shape[axes[1]]
    out_shape = x.shape[axes[0]], x.shape[axes[1]]
    back = ifft2((fft2(x, axes=axes)) * fft2(k, axes=axes, s=out_shape), axes=axes)
    return back.real[kh - 1 :, kw - 1 :]


def corr_norm_mask_match(x: mat_i2, tmpl: mat_i2, mask: mat_u1):
    r"""``TM_CCORR_NORMED`` mode in opecv matchTemplate.

    :param x: data mat. dtype should be int/uint.
    :param tmpl: template mat. dtype should be the same with that of `x`.
    :param mask: a binary mask mat in ``{0, 1}``. dtype should be int/uint.

    .. math::

        R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y') \cdot M(x',y')^2)}{\sqrt{\sum_{x',y'} \left( T(x',y') \cdot M(x',y') \right)^2 \cdot \sum_{x',y'} \left( I(x+x',y+y') \cdot M(x',y') \right)^2}}
    """
    mask_template = mask * tmpl
    mask_template = np.flip(mask_template, axis=(0, 1))
    mask_template = mask_template / np.linalg.norm(mask_template, axis=(0, 1))
    corr = conv2d(x, mask_template)

    # NOTE: suppose x is in (-(2^8 - 1), 2^8 - 1), saved in int16
    # then x^2 is in (0, 2^16-1), cannot be saved in int16 but can be saved in uint16.
    # so we should cast |x| into uint16 and calc. |x|^2, which is in uint16.
    abs_x = np.abs(x).astype(np.dtype(f"u{x.dtype.itemsize}"))

    mask = np.flip(mask, axis=(0, 1))
    img_norm = np.sqrt(conv2d(np.power(abs_x, 2), mask))
    img_norm[img_norm < 1e-12] = +np.inf
    return (corr / img_norm).sum(axis=-1)


def hdiff_corr_norm_mask_match(x: mat_u1, tmpl: mat_u1, mask: mat_u1):
    x = x.astype(np.int16)
    tmpl = tmpl.astype(np.int16)
    x_diff: mat_i2 = x[:, 1:] - x[:, :-1]  # type: ignore
    t_diff: mat_i2 = tmpl[:, 1:] - tmpl[:, :-1]  # type: ignore
    mask_u = mask[:, 1:] * mask[:, :-1]
    return corr_norm_mask_match(x_diff, t_diff, mask_u)


def vdiff_corr_norm_mask_match(x: mat_u1, tmpl: mat_u1, mask: mat_u1):
    x = x.astype(np.int16)
    tmpl = tmpl.astype(np.int16)
    y_diff: mat_i2 = x[1:] - x[:-1]  # type: ignore
    t_diff: mat_i2 = tmpl[1:] - tmpl[:-1]  # type: ignore
    mask_u = mask[1:] * mask[:-1]
    return corr_norm_mask_match(y_diff, t_diff, mask_u)


def energy_mask(x: mat_u1, tmpl: mat_u1, mask: mat_u1, threshold=0.6):
    mask_template = mask * tmpl
    x_u2 = x.astype(np.uint16)
    tpl_energy = np.linalg.norm(mask_template, axis=(0, 1))

    mask = np.flip(mask, axis=(0, 1))
    img_sqe = np.sqrt(conv2d(np.power(x_u2, 2), mask))
    return np.any(img_sqe > (threshold * tpl_energy), axis=-1)


def sdiff_mask_match(x: mat_u1, tmpl: mat_u1, mask: mat_u1):
    mask_template = mask * tmpl
    mask_template = np.flip(mask_template, axis=(0, 1))
    corr = conv2d(x, mask_template)

    x_u2 = x.astype(np.uint16)
    mask = np.flip(mask, axis=(0, 1))
    img_energy = conv2d(np.power(x_u2, 2), mask)

    return (img_energy - 2 * corr).sum(axis=-1)
