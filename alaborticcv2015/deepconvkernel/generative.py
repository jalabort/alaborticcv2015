from __future__ import division
import numpy as np
import warnings
from menpo.math import pca, ica, nmf
from menpo.math.decomposition.ica import _batch_ica, negentropy_logcosh
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from .base import LearnableLDCN, _normalize_images


def learn_pca_filters(patches, n_filters=8):
    r"""
    Learn PCA convolution filters
    """
    n_patches = patches.shape[0]
    patch_shape = patches.shape[-3:]
    patches = patches.reshape((n_patches, -1))
    # learn pca filters
    pca_filters = pca(patches, inplace=True)[0]
    pca_filters = pca_filters[:n_filters, :]
    return pca_filters.reshape((-1,) + patch_shape)


def learn_ica_filters(patches, n_filters=8, algorithm=_batch_ica,
                      negentropy=negentropy_logcosh, max_iters=500,
                      verbose=False):
    r"""
    Learn ICA convolution filters
    """
    n_patches = patches.shape[0]
    patch_shape = patches.shape[-3:]
    patches = patches.reshape((n_patches, -1))
    # learn ica filters
    ica_filters = ica(patches, algorithm=algorithm,
                      negentropy_approx=negentropy, n_components=n_filters,
                      max_iters=max_iters, inplace=True, verbose=verbose)[0]
    return ica_filters.reshape((-1,) + patch_shape)


def learn_nmf_filters(patches, n_filters=8, max_iters=500, verbose=False,
                      **kwargs):
    r"""
    Learn NMF convolution filters
    """
    n_patches = patches.shape[0]
    patch_shape = patches.shape[-3:]
    patches = patches.reshape((n_patches, -1))
    # learn nmf filters
    nmf_filters = nmf(patches, n_components=n_filters, max_iters=max_iters,
                      verbose=verbose, **kwargs)
    return nmf_filters.reshape((-1,) + patch_shape)


def _parse_params(learn_filters, n_filters, n_layers):
    if hasattr(learn_filters, '__call__'):
        if isinstance(n_filters, int):
            return [[learn_filters]] * n_layers, [[n_filters]] * n_layers
        elif isinstance(n_filters, list):
            if len(n_filters) != n_layers:
                warnings.warn('n_layers does not agree with n_filters, '
                              'the number of levels will be set based on '
                              'n_filters.')
            return [[learn_filters]] * len(n_filters), [[n] for n in n_filters]
    elif isinstance(learn_filters, list):
        if len(learn_filters) == 1:
            return _parse_params(learn_filters[0], n_filters, n_layers)
        elif hasattr(learn_filters[0], '__call__'):
            if isinstance(n_filters, int):
                return [learn_filters] * n_layers, \
                       [[n_filters] * len(learn_filters)] * n_layers
            elif isinstance(n_filters, list):
                if len(n_filters) != n_layers:
                    warnings.warn('n_layers does not agree with n_filters, '
                                  'the number of levels will be set '
                                  'based on n_filters.')
                    n_layers = len(n_filters)
                n_list = []
                for j, nfs in enumerate(n_filters):
                    if isinstance(nfs, int) or len(nfs) == 1:
                        n_list.append([nfs] * len(learn_filters))
                    elif len(nfs) == len(learn_filters):
                        n_list.append(nfs)
                    else:
                        raise ValueError('n_filters does not agree with '
                                         'learn_filters. Position {} of '
                                         'n_filters must be and integer, '
                                         'a list containing a single integer '
                                         'or a list containing the same '
                                         'number of integers as the length of'
                                         'learn_filters.').format(j)
                return [learn_filters] * n_layers,  n_list
        elif isinstance(learn_filters[0], list):
            if len(learn_filters) != n_layers:
                warnings.warn('n_layers does not agree with learn_filters, '
                              'the number of levels will be set based on '
                              'learn_filters.')
            if len(learn_filters) == len(n_filters):
                learn_list = []
                n_list = []
                for j, (lfs, nfs) in enumerate(zip(learn_filters, n_filters)):
                    if isinstance(nfs, int) or len(nfs) == 1:
                        lfs, nfs = _parse_params(lfs, nfs, 1)
                        lfs = lfs[0]
                        nfs = nfs[0]
                    elif len(lfs) != len(nfs):
                        raise ValueError('n_filters does not agree with '
                                         'learn_filters. The total number of '
                                         'elements in sublist {} of both '
                                         'lists is not the same.').format(j)
                    learn_list.append(lfs)
                    n_list.append(nfs)
                return learn_list, n_list
            else:
                raise ValueError('n_filters does not agree with learn_filters.'
                                 ' The total number of elements in both '
                                 'lists is not the same.')


class GenerativeLDCN(LearnableLDCN):
    r"""
    Generative Linear Deep Convolutional Network Class
    """
    def __init__(self, learn_filters=learn_pca_filters, n_filters=8,
                 n_layers=3, architecture=3, normalize_patches=centralize,
                 normalize_filters=None, patch_shape=(7, 7), mode='same',
                 boundary='constant'):
        super(GenerativeLDCN, self).__init__(architecture=architecture,
                                             patch_shape=patch_shape,
                                             mode=mode, boundary=boundary)
        self._learn_filters, self._n_filters = _parse_params(
            learn_filters, n_filters, n_layers)
        self.normalize_patches = normalize_patches
        self.normalize_filters = normalize_filters

    def _learn_network(self, images, extract_patches_func, verbose=False,
                       **kwargs):
        if verbose:
            print_dynamic('- Learning network\n')

        # # convert images to the appropriate type
        # convert_images_to_dtype_inplace(images, dtype=self.dtype)

        level_str = ''
        n_ch = images[0].n_channels
        filters = []
        for l, (lfs, nfs) in enumerate(zip(self._learn_filters,
                                           self._n_filters)):
            if verbose:
                if self.n_layers > 1:
                    level_str = '  - Layer {}: '.format(l)
                else:
                    level_str = '  - '
            # extract patches
            patches = extract_patches_func(images, string=level_str)
            if verbose:
                string = level_str + 'Learning filters '
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(0, show_bar=True)))
            # normalize patches
            if self.normalize_patches:
                patches = _normalize_images(patches,
                                            norm_func=self.normalize_patches)
            # learn level filters
            fs = np.empty((self.n_filters_layer[l], n_ch) + self.patch_shape)
            start = 0
            for (lf, nf) in zip(lfs, nfs):
                fs[start:start+nf] = lf(patches, nf, **kwargs)
                start += nf
            # delete patches
            del patches
            # normalize filters
            if self.normalize_filters:
                fs = _normalize_images(fs, norm_func=self.normalize_filters)
            # save filters
            filters.append(fs)
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(1, show_bar=True)))
            if l < self.n_layers-1:
                # if not last layer, compute responses
                images = self.compute_filters_responses(
                    images, fs, norm_func=self.normalize_filters,
                    mode=self.mode, boundary=self.boundary, verbose=verbose,
                    string=level_str)
                n_ch = images[0].n_channels
            if verbose:
                print_dynamic('{}Done!\n'.format(level_str))
        self._filters = filters
