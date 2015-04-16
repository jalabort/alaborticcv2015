from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from menpo.math import mccf, lda
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from .base import LearnableLDCN, _parse_filters, _normalize_images


def _build_grid(shape):
    shape = np.array(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    start = -half_shape
    end = half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)


def _generate_gaussian_response(shape, cov):
    mvn = multivariate_normal(mean=np.zeros(2), cov=cov)
    grid = _build_grid(shape)
    return mvn.pdf(grid)[None]


def learn_mccf_filters(patches, patch_shape=(7, 7), cov=3, l=0.01,
                       boundary='constant', verbose=False, level_str=''):
    r"""
    Learn MCCF convolution filters
    """
    if verbose:
        level_str += 'Learning filters '
    n_filters = patches.shape[1]
    n_channels = patches.shape[2]
    # generate desired response
    response = _generate_gaussian_response(patch_shape, cov)
    # learn mccf filters
    mccf_filters = np.empty((n_filters, n_channels) + patch_shape)
    for j in range(n_filters):
        if verbose:
            print_dynamic('{}{}'.format(
                level_str, progress_bar_str(j/n_filters, show_bar=True)))
        mccf_filters[j] = mccf(patches[:, j], response, l=l,
                               boundary=boundary)[0]
    return mccf_filters


def learn_lda_filters(patches, n_filters=None, verbose=False, level_str='',
                      **kwargs):
    r"""
    Learn LDA convolution filters
    """
    if verbose:
        level_str += 'Learning filters '
    n_patches, n_classes = patches.shape[:2]
    patch_shape = patches.shape[-3:]
    if n_filters is None:
        n_filters = np.minimum(n_classes, np.prod(patch_shape)) - 1
    # reshape patches
    patches = np.rollaxis(patches, 1, 0)
    patches = patches.reshape((n_patches * n_classes, -1))
    # generate class offsets
    class_offsets = [n_patches * j for j in range(1, n_classes)]
    # learn lda filters
    lda_filters = lda(patches, class_offsets, n_components=n_filters,
                      inplace=True)[0]
    return lda_filters.reshape((-1,) + patch_shape)


class DiscriminativeLDCN(LearnableLDCN):
    r"""
    Discriminative Linear Deep Convolutional Network Class
    """
    def __init__(self, learn_filters=learn_mccf_filters, n_layers=3,
                 architecture=3, norm_func=centralize, patch_shape=(31, 31),
                 mode='same', boundary='constant'):
        super(DiscriminativeLDCN, self).__init__(architecture=architecture,
                                                 norm_func=norm_func,
                                                 patch_shape=patch_shape,
                                                 mode=mode, boundary=boundary)
        self._learn_filters = learn_filters
        self._n_layers = n_layers
        print learn_filters
        if learn_filters is learn_mccf_filters:
            self.context_shape = self.patch_shape * 3

    def learn_network_from_grid(self, images, stride=4, verbose=False,
                                **kwargs):
        raise Exception('method learn_network_from_grid is not available on '
                        'DiscriminativeLDCN.')

    def _learn_network(self, images, extract_patches_func, verbose=False,
                       **kwargs):
        if verbose:
            print_dynamic('- Learning network\n')

        # # convert images to the appropriate type
        # convert_images_to_dtype_inplace(images, dtype=self.dtype)

        level_str = ''
        filters = []
        for l in range(self._n_layers):
            if verbose:
                if self._n_layers > 1:
                    level_str = '  - Layer {}: '.format(l)
                else:
                    level_str = '  - '
            # extract patches
            patches = extract_patches_func(images, flatten=False,
                                           string=level_str)
            if verbose:
                string = level_str + 'Learning filters '
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(0, show_bar=True)))
            # normalize patches
            patches = _normalize_images(patches, norm_func=self.norm_func)
            # learn level filters
            fs = self._learn_filters(patches, patch_shape=self.patch_shape,
                                     verbose=verbose, string=level_str,
                                     **kwargs)
            # delete patches
            del patches
            # normalize filters
            fs = _normalize_images(fs, norm_func=self.norm_func)
            # save filters
            filters.append(fs)
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(1, show_bar=True)))
            if l < self._n_layers-1:
                # if not last layer, compute responses
                images = self.compute_filters_responses(
                    images, fs, norm_func=self.norm_func, mode=self.mode,
                    boundary=self.boundary, verbose=verbose, string=level_str)
            if verbose:
                print_dynamic('{}Done!\n'.format(level_str))
        self._filters = filters
        self._n_filters = _parse_filters(filters)

