from __future__ import division
import numpy as np
from numpy.fft import ifft2, fftshift
import warnings
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.deepconvkernel.base import (
    _network_response, _parse_filters)
from alaborticcv2015.utils import (
    pad, fft_convolve2d_sum,
    extract_patches,
    extract_patches_from_grid, extract_patches_from_landmarks,
    centralize)
from .base import LinDeepConvNet


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
            return parse_gabor_params(params[0], n_layers)
        else:
            if len(params) != n_layers:
                warnings.warn('n_layers does not agree with params, '
                              'the number of levels will be set based on '
                              'params.')
            n_filters = []
            for p in params:
                n_filters.append(p['num_scales'] * p['num_orientations'])
            return params, n_filters


def _parse_stride(stride):
    if len(stride) == 1:
        return stride, stride
    if len(stride) == 2:
        return stride


def normalize_patches(patches, norm_func=centralize):
    for j, p in enumerate(patches):
        patches[j] = norm_func(p)
    return patches


def learn_network_from_grid(images, n_layers, patch_shape=(7, 7),
                            stride=(4, 4), norm_func=centralize,
                            verbose=False):
    def extract_patches_func(imgs):
        return extract_patches(imgs, extract=extract_patches_from_grid,
                               patch_shape=patch_shape, stride=stride)
    return _learn_network(images, extract_patches_func, n_layers=n_layers,
                          norm_func=norm_func, verbose=verbose)


def learn_network_from_landmarks(images, n_layers, patch_shape=(7, 7),
                                 group=None, label=None, norm_func=centralize,
                                 verbose=False):
    def extract_patches_func(imgs):
        return extract_patches(imgs, extract=extract_patches_from_landmarks,
                               patch_shape=patch_shape, group=group,
                               label=label)
    return _learn_network(images, extract_patches_func, n_layers=n_layers,
                          norm_func=norm_func, verbose=verbose)


def _learn_network(image, extract_patches_func, n_layers=3,
                   norm_func=centralize, verbose=False):
        if verbose:
            string = '- Learning network'

        images = [image]
        filters = []
        for l in range(n_layers):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(l/n_layers, show_bar=True)))

            # learn filters
            fs = extract_patches_func(images)
            if norm_func is not None:
                # normalize filters if required
                fs = normalize_patches(fs, norm_func=norm_func)
            # compute responses
            images = _compute_filter_responses(images, fs)
            # save filters
            filters.append(fs)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

        return filters


def _compute_filter_responses(images, filters, hidden_mode='same',
                              visible_mode='valid', boundary='symmetric'):
        return [_network_response(i, filters, hidden_mode=hidden_mode,
                                  visible_mode=visible_mode, boundary=boundary)
                for i in images]


class SelfSimLDCN(LinDeepConvNet):
    r"""
    Self-Similarity Linear Deep Convolutional Network
    """
    def __init__(self, n_layers=3, patch_shape=(7, 7),
                 norm_func=centralize, verbose=False):
        self._n_layers = n_layers
        self.patch_shape = patch_shape
        self.norm_func = norm_func
        self.verbose = verbose

    def learn_network_from_grid(self, image, stride=(4, 4)):
        stride = _parse_stride(stride)
        self._filters = learn_network_from_grid(
            image, n_layers=self._n_layers, patch_shape=self.patch_shape,
            stride=stride, norm_func=self.norm_func, verbose=self.verbose)
        self._n_filters = _parse_filters(self._filters)

    def learn_network_from_landmarks(self, image, group=None, label=None):
        self._filters = learn_network_from_landmarks(
            image, n_layers=self._n_layers, patch_shape=self.patch_shape,
            group=group, label=label, norm_func=self.norm_func,
            verbose=self.verbose)
        self._n_filters = _parse_filters(self._filters)
