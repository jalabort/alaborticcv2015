from __future__ import division
import numpy as np
from numpy.fft import ifft2, fftshift
import warnings
from menpo.math import log_gabor
from menpo.feature import centralize
from .base import LinDeepConvNet, normalize_filters


def _parse_params(params, n_layers):
    if params is None:
        params = {'num_scales': 4,
                  'num_orientations': 6,
                  'min_wavelength': 3,
                  'scaling_constant': 2,
                  'center_sigma': 0.65,
                  'd_phi_sigma': 1.3}

    if isinstance(params, dict):
            n_filters = params['num_scales'] * params['num_orientations']
            return [params] * n_layers, [n_filters] * n_layers
    elif isinstance(params, list):
        if len(params) == 1:
            return _parse_params(params[0], n_layers)
        else:
            if len(params) != n_layers:
                warnings.warn('n_layers does not agree with params, '
                              'the number of levels will be set based on '
                              'params.')
            n_filters = []
            for p in params:
                n_filters.append(p['num_scales'] * p['num_orientations'])
            return params, n_filters


class LogGaborLDCN(LinDeepConvNet):
    r"""
    Log-Gabor Linear Deep Convolutional Network Class
    """
    def __init__(self, params=None, n_layers=3, patch_shape=(7, 7),
                 norm_func=centralize):
        self.params, self._n_filters = _parse_params(params, n_layers)
        self.patch_shape = patch_shape
        self.norm_func = norm_func

    def build_network(self, n_channels=3):
        filters = []
        n_channels_layer = [n_channels] + self.n_filters_layer[:-1]
        for gp, n_ch in zip(self.params, n_channels_layer):
            fs = log_gabor(np.empty(self.patch_shape),
                           num_scales=gp['num_scales'],
                           num_orientations=gp['num_orientations'],
                           min_wavelength=gp['min_wavelength'],
                           scaling_constant=gp['scaling_constant'],
                           center_sigma=gp['center_sigma'],
                           d_phi_sigma=gp['d_phi_sigma'])[3]
            fs = np.real(fftshift(ifft2(fs), axes=(-2, -1)))
            fs = np.tile(fs[:, None, ...], (1, n_ch, 1, 1))
            filters.append(fs)

        self._filters = normalize_filters(filters, self.norm_func)
