from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from menpo.image import Image


def pad(pixels, ext_shape, mode='constant'):
    _, h, w = pixels.shape

    h_margin = (ext_shape[0] - h) // 2
    w_margin = (ext_shape[1] - w) // 2

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[0]:
        h_margin += 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[1]:
        w_margin += 1

    pad_width = ((0, 0), (h_margin, h_margin2), (w_margin, w_margin2))

    return np.lib.pad(pixels, pad_width, mode=mode)


def crop(pixels, shape):
    _, h, w = pixels.shape

    h_margin = (h - shape[0]) // 2
    w_margin = (w - shape[1]) // 2

    h_corrector = 1 if np.remainder(h - shape[0], 2) != 0 else 0
    w_corrector = 1 if np.remainder(w - shape[1], 2) != 0 else 0

    return pixels[:,
                  h_margin + h_corrector:-h_margin,
                  w_margin + w_corrector:-w_margin]


def conv(i, f, mode='same', boundary='constant'):
    # extended shape
    i_shape = np.asarray(i.shape)
    f_shape = np.asarray(f.shape)
    ext_shape = i_shape + f_shape - 1

    # extend image and filter
    ext_i = pad(i.pixels, ext_shape, mode=boundary)
    ext_f = pad(f.pixels, ext_shape)

    # compute ffts of extended image and extended filter
    fft_ext_i = fft2(ext_i)
    fft_ext_f = fft2(ext_f)

    # compute extended convolution in Fourier domain
    fft_ext_c = fft_ext_f * fft_ext_i

    # compute ifft of extended convolution
    ext_c = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        c = ext_c
    elif mode is 'same':
        c = crop(ext_c, i_shape)
    elif mode is 'valid':
        c = crop(ext_c, i_shape - f_shape + 1)

    return Image(c)


def multiconvsum(i, fs, mode='same', boundary='constant'):
    # extended shape
    i_shape = np.asarray(i.shape)
    f_shape = np.asarray(fs[0].shape)
    ext_shape = i_shape + f_shape - 1

    # extend image
    ext_i = pad(i.pixels, ext_shape, mode=boundary)
    # compute ffts of extended image
    fft_ext_i = fft2(ext_i)

    # initialize extended response shape
    ext_r = np.empty(np.hstack((len(fs), ext_shape)))
    for j, f in enumerate(fs):
        # extend filter
        ext_f = pad(f.pixels, ext_shape)
        # compute fft of filter
        fft_ext_f = fft2(ext_f)

        # compute extended convolution in Fourier domain
        fft_ext_c = np.sum(fft_ext_f * fft_ext_i, axis=0)

        # compute ifft of extended convolution
        ext_r[j] = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        r = ext_r
    elif mode is 'same':
        r = crop(ext_r, i_shape)
    elif mode is 'valid':
        r = crop(ext_r, i_shape - f_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))

    return Image(r)


def multiconvlist(i, fs, mode='same', boundary='constant'):
    # extended shape
    i_shape = np.asarray(i.shape)
    f_shape = np.asarray(fs[0].shape)
    ext_shape = i_shape + f_shape - 1

    # extend image
    ext_i = pad(i.pixels, ext_shape, mode=boundary)
    # compute ffts of extended image
    fft_ext_i = fft2(ext_i)

    # initialize responses list
    rs = []
    for j, f in enumerate(fs):
        # extend filter
        ext_f = pad(f.pixels, ext_shape)
        # compute fft of filter
        fft_ext_f = fft2(ext_f)

        # compute extended convolution in Fourier domain
        fft_ext_c = fft_ext_f * fft_ext_i

        # compute ifft of extended convolution
        ext_r = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

        if mode is 'full':
            r = ext_r
        elif mode is 'same':
            r = crop(ext_r, i_shape)
        elif mode is 'valid':
            r = crop(ext_r, i_shape - f_shape + 1)
        else:
            raise ValueError(
                "mode={}, is not supported. The only supported "
                "modes are: 'full', 'same' and 'valid'.".format(mode))
        # add
        rs.append(Image(r))

    return rs
