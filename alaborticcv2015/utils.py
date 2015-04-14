from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from functools import wraps
from menpo.feature import ndfeature
from menpo.feature.base import rebuild_feature_image
from menpo.shape import PointCloud

# Cache the greyscale luminosity coefficients as they are invariant.
_greyscale_luminosity_coef = None


def pad(pixels, ext_shape, boundary='constant'):
    h, w = pixels.shape[-2:]

    h_margin = (ext_shape[0] - h) // 2
    w_margin = (ext_shape[1] - w) // 2

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[0]:
        h_margin += 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[1]:
        w_margin += 1

    pad_width = []
    for _ in pixels.shape[:-2]:
        pad_width.append((0, 0))
    pad_width += [(h_margin, h_margin2), (w_margin, w_margin2)]
    pad_width = tuple(pad_width)

    return np.lib.pad(pixels, pad_width, mode=boundary)


def crop(pixels, shape):
    h, w = pixels.shape[-2:]

    h_margin = (h - shape[0]) // 2
    w_margin = (w - shape[1]) // 2

    h_corrector = 1 if np.remainder(h - shape[0], 2) != 0 else 0
    w_corrector = 1 if np.remainder(w - shape[1], 2) != 0 else 0

    return pixels[...,
                  h_margin + h_corrector:-h_margin,
                  w_margin + w_corrector:-w_margin]


def ndconvolution(wrapped):

    @wraps(wrapped)
    def wrapper(image, filter, *args, **kwargs):
        if not isinstance(image, np.ndarray) and not isinstance(filter, np.ndarray):
            # Both image and filter are menpo images
            feature = wrapped(image.pixels, filter.pixels, *args, **kwargs)
            return rebuild_feature_image(image, feature)
        elif not isinstance(image, np.ndarray):
            # Image is menpo image
            feature = wrapped(image.pixels, filter, *args, **kwargs)
            return rebuild_feature_image(image, feature)
        elif not isinstance(filter, np.ndarray):
            # filter is menpo image
            return wrapped(image, filter, *args, **kwargs)
        else:
            return wrapped(image, filter, *args, **kwargs)
    return wrapper


@ndconvolution
def fft_convolve2d(x, f, mode='same', boundary='constant'):
    r"""
    Performs fast 2d convolution in the frequency domain convolving each image
    channel with its corresponding filter channel.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Image.
    f : ``(channels, height, width)`` `ndarray`
        Filter.
    mode : str {`full`, `same`, `valid`}, optional
        Determines the shape of the resulting convolution.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    Returns
    -------
    c: ``(channels, height, width)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel.
    """
    # extended shape
    x_shape = np.asarray(x.shape[-2:])
    f_shape = np.asarray(f.shape[-2:])
    ext_shape = x_shape + f_shape - 1

    # extend image and filter
    ext_x = pad(x, ext_shape, boundary=boundary)
    ext_f = pad(f, ext_shape)

    # compute ffts of extended image and extended filter
    fft_ext_x = fft2(ext_x)
    fft_ext_f = fft2(ext_f)

    # compute extended convolution in Fourier domain
    fft_ext_c = fft_ext_f * fft_ext_x

    # compute ifft of extended convolution
    ext_c = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, x_shape)
    elif mode is 'valid':
        return crop(ext_c, x_shape - f_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))


@ndconvolution
def fft_convolve2d_sum(x, f, mode='same', boundary='constant', axis=0,
                       keepdims=False):
    r"""
    Performs fast 2d convolution in the frequency domain convolving each image
    channel with its corresponding filter channel and summing across the
    channel axis.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Image.
    f : ``(channels, height, width)`` `ndarray`
        Filter.
    mode : str {`full`, `same`, `valid`}, optional
        Determines the shape of the resulting convolution.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    axis : `int`, optional
        The axis across to which the summation is performed.
    keepdims: `boolean`, optional
        If `True` the number of dimensions of the result is the same as the
        number of dimensions of the filter. If `False` the channel dimension
        is lost in the result.
    Returns
    -------
    c: ``(1, height, width)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel and summing across the channel axis.
    """
    return np.sum(fft_convolve2d(x, f, mode=mode, boundary=boundary),
                  axis=axis, keepdims=keepdims)


@ndfeature
def convert_image_to_dtype_inplace(pixels, dtype=None):
    if dtype:
        pixels = pixels.astype(dtype=dtype)
    return pixels


def extract_patches_from_landmarks(image, patch_shape=(7, 7), group=None,
                                   label=None, as_single_array=False,
                                   dtype=np.float32):
    image_cp = image.copy()
    image_cp.pixels_astype(dtype=np.float64)
    patches = image_cp.extract_patches_around_landmarks(
        group=group, label=label, patch_size=patch_shape,
        as_single_array=as_single_array)[:, 0, ...]
    del image_cp
    for j, p in enumerate(patches):
        patches[j] = convert_image_to_dtype_inplace(p, dtype=dtype)
    return patches


def extract_patches_from_grid(image, patch_shape=(7, 7), stride=(4, 4),
                              as_single_array=False, dtype=np.float32):
    limit = np.round(np.asarray(patch_shape) / 2)
    image_cp = image.copy()
    image_cp.pixels_astype(dtype=np.float64)
    h, w = image_cp.shape[-2:]
    grid = np.rollaxis(np.mgrid[limit[0]:h-limit[0]:stride[0],
                                limit[1]:w-limit[0]:stride[1]], 0, 3)
    pc = PointCloud(np.require(grid.reshape((-1, 2)), dtype=np.float64))
    patches = image_cp.extract_patches(pc, patch_size=patch_shape,
                                       as_single_array=as_single_array)
    del image_cp
    for j, p in enumerate(patches):
        patches[j] = convert_image_to_dtype_inplace(p, dtype=dtype)
    return patches


def extract_patches(images, extract=extract_patches_from_grid,
                    patch_shape=(7, 7), as_single_array=False,
                    dtype=np.float32, **kwargs):
    if not as_single_array:
        patches = []
        for i in images:
            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            patches.append(ps.reshape((-1,) + ps.shape[-3:]))
    else:
        ps = extract(images[0], patch_shape=patch_shape, dtype=dtype,
                     as_single_array=as_single_array, **kwargs)
        patches = ps.reshape((-1,) + ps.shape[-3:])
        for i in images:
            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            ps = ps.reshape((-1,) + ps.shape[-3:])
            patches = np.concatenate((patches, ps), axis=0)
    return patches


@ndfeature
def centralize(x, channels=True, dtype=None):
    if channels:
        m = np.mean(x, axis=(-2, -1))[..., None, None]
    else:
        m = np.mean(x)
    x = x - m
    if dtype:
        x = x.astype(dtype=dtype)
    return x


@ndfeature
def normalize_norm(x, channels=True, dtype=None):
    x = centralize(x, channels=channels, dtype=dtype)
    if channels:
        n = np.asarray(np.linalg.norm(x, axis=(-2, -1))[..., None, None])
    else:
        n = np.asarray(np.linalg.norm(x))
    x = x / n
    if dtype:
        x = x.astype(dtype=dtype)
    return x


@ndfeature
def normalize_std(x, channels=True, dtype=None):
    x = centralize(x, channels=channels, dtype=dtype)
    if channels:
        n = np.std(x, axis=(-2, -1))[..., None, None]
    else:
        n = np.std(x)
    x = x / n
    if dtype:
        x = x.astype(dtype=dtype)
    return x


@ndfeature
def greyscale(x, mode='luminosity', channel=None):
    # Only compute the coefficients once.
    if mode == 'luminosity':
        global _greyscale_luminosity_coef
        if _greyscale_luminosity_coef is None:
            _greyscale_luminosity_coef = np.linalg.inv(
                np.array([[1.0, 0.956, 0.621],
                          [1.0, -0.272, -0.647],
                          [1.0, -1.106, 1.703]]))[0, :]
        x = np.einsum('i,ikl->kl', _greyscale_luminosity_coef, x)
    elif mode == 'average':
        x = np.mean(x, axis=0)
    elif mode == 'channel':
        if channel is None:
            raise ValueError("For the 'channel' mode you have to provide"
                             " a channel index")
        x = x[channel, ...]
    else:
        raise ValueError("Unknown mode {} - expected 'luminosity', "
                         "'average' or 'channel'.".format(mode))
    return x[None, ...]


def normalize_patches(patches, norm_func=centralize):
    if norm_func is not None:
        for j, p in enumerate(patches):
            patches[j] = norm_func(p)
    return patches
