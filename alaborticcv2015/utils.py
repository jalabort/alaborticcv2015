from __future__ import division
import numpy as np


def pad(pixels, ext_shape, mode='constant'):
    _, h, w = pixels.shape

    h_margin = np.floor((ext_shape[0] - h) / 2)
    w_margin = np.floor((ext_shape[1] - w) / 2)

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[0]:
        h_margin2 = h_margin + 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[1]:
        w_margin2 = w_margin + 1

    pad_width = ((0, 0), (h_margin, h_margin2), (w_margin, w_margin2))

    return np.lib.pad(pixels, pad_width, mode=mode)


def unpad(pixels, red_shape):
    _, h, w = pixels.shape

    h_margin = (h - red_shape[0]) // 2
    w_margin = (w - red_shape[1]) // 2

    h_margin2 = h_margin
    if np.remainder(h - red_shape[0], 2) != 0:
        h_margin2 += 1

    w_margin2 = w_margin
    if np.remainder((w - red_shape[1]), 2) != 0:
        w_margin2 += 1

    return pixels[:, h_margin:-h_margin2, w_margin:-w_margin2]
