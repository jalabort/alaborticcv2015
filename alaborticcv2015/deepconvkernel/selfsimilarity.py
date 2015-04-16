from __future__ import division
import warnings
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.deepconvkernel.base import _parse_filters
from .base import LearnableLDCN, _normalize_images


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
    def __init__(self, n_layers=3, architecture=3, norm_func=centralize,
                 patch_shape=(7, 7),  mode='same', boundary='constant'):
        super(SelfSimLDCN, self).__init__(architecture=architecture,
                                          norm_func=norm_func,
                                          patch_shape=patch_shape,
                                          mode=mode, boundary=boundary)
        self._n_layers = n_layers

    def _learn_network(self, image, extract_patches_func, verbose=False,
                       **kwargs):
        if verbose:
            print_dynamic('- Learning network\n')
        level_str = ''
        images = [image]
        filters = []
        for l in range(self._n_layers):
            if verbose:
                if self._n_layers > 1:
                    level_str = '  - Layer {}: '.format(l)
                else:
                    level_str = '  - '
            # extract patches/filters
            fs = extract_patches_func(images, string=level_str)
            if verbose:
                string = level_str + 'Learning filters '
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(0, show_bar=True)))
            # flip filters
            fs = fs[..., ::-1, ::-1]
            # normalize filters
            fs = _normalize_images(fs, norm_func=self.norm_func)
            # save filters
            filters.append(fs)
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(1, show_bar=True)))
            if l != self.n_layers:
                # compute responses if not last layer
                images = self.compute_filters_responses(
                    images, fs, norm_func=self.norm_func, mode=self.mode,
                    boundary=self.boundary, verbose=verbose, string=level_str)
            if verbose:
                print_dynamic('{}Done!\n'.format(level_str))
        self._filters = filters
        self._n_filters = _parse_filters(filters)

