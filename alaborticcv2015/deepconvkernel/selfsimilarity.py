from __future__ import division
import warnings
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.deepconvkernel.base import _parse_filters
from alaborticcv2015.utils import normalize_patches
from .base import LearnableLDCN, compute_filters_responses


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


class SelfSimLDCN(LearnableLDCN):
    r"""
    Self-Similarity Linear Deep Convolutional Network Class
    """
    def __init__(self, n_layers=3, patch_shape=(7, 7),
                 norm_func=centralize, verbose=False):
        self._n_layers = n_layers
        self.patch_shape = patch_shape
        self.norm_func = norm_func
        self.verbose = verbose

    def _learn_network(self, image, extract_patches_func, verbose=False,
                       **kwargs):
        if verbose:
            string = '- Learning network'

        images = [image]
        filters = []
        for l in range(self._n_layers):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(l/self._n_layers, show_bar=True)))

            # extract patches/filters
            fs = extract_patches_func(images)
            # flip filters
            fs = fs[..., ::-1, ::-1]
            # normalize filters
            fs = normalize_patches(fs, norm_func=self.norm_func)
            # save filters
            filters.append(fs)
            if l != self.n_layers:
                # compute responses if not last layer
                images = compute_filters_responses(images, fs,
                                                   norm_func=self.norm_func)

        self._filters = filters
        self._n_filters = _parse_filters(filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))
