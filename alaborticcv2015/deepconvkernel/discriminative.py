from __future__ import division
import numpy as np
import warnings
from scipy.stats import multivariate_normal
from sklearn import svm
from menpo.math import mccf, lda
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from .base import (
    LearnableLDCN, _parse_filters, _normalize_images,
    _extract_patches, _extract_patches_from_landmarks)


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


def _parse_n_filters(n_filters, n_layers):
    if isinstance(n_filters, int):
        return [n_filters] * n_layers, n_layers
    elif isinstance(n_filters, list):
        if len(n_filters) != n_layers:
            warnings.warn('n_layers does not agree with n_filters, '
                          'the number of levels will be set based on '
                          'n_filters.')
        return n_filters, len(n_filters)


def learn_lda_filters(class_patches, n_filters=None, verbose=False,
                      level_str='', **kwargs):
    r"""
    Learn LDA convolution filters
    """
    if verbose:
        level_str += 'Learning filters '
    if isinstance(class_patches, list):
        patch_shape = class_patches[0].shape[-3:]
        n_patches = 0
        for patches in class_patches:
            n_patches += patches.shape[0]
        class_patches_array = np.empty((n_patches, np.prod(patches.shape[1:])))
        class_offsets = []
        start = 0
        for patches in class_patches:
            finish = start + patches.shape[0]
            class_offsets.append(finish)
            class_patches_array[start:finish] = patches.reshape((
                patches.shape[0], -1))
            start = finish
        class_patches = class_patches_array
        class_offsets = class_offsets[:-1]
    else:
        n_patches, n_classes = class_patches.shape[:2]
        patch_shape = class_patches.shape[-3:]
        if n_filters is None:
            n_filters = np.minimum(n_classes, np.prod(patch_shape)) - 1
        # reshape patches
        class_patches = np.rollaxis(class_patches, 1, 0)
        class_patches = class_patches.reshape((n_patches * n_classes, -1))
        # generate class offsets
        class_offsets = [n_patches * j for j in range(1, n_classes)]

    # learn lda filters
    lda_filters = lda(class_patches, class_offsets, n_components=n_filters,
                      inplace=True)[0]
    return lda_filters.reshape((-1,) + patch_shape)[..., ::-1, ::-1]


def learn_svm_filters(class_patches, verbose=False, level_str='', **kwargs):
    r"""
    Learn SVM convolution filters
    """
    if verbose:
        level_str += 'Learning filters '
    if isinstance(class_patches, list):
        patch_shape = class_patches[0].shape[-3:]
        n_patches = 0
        for patches in class_patches:
            n_patches += patches.shape[0]
        class_patches_array = np.empty((n_patches, np.prod(patches.shape[1:])))
        class_offsets = np.empty(n_patches)
        start = 0
        for j, patches in enumerate(class_patches):
            finish = start + patches.shape[0]
            class_offsets[start:finish] = j
            class_patches_array[start:finish] = patches.reshape((
                patches.shape[0], -1))
            start = finish
        class_patches = class_patches_array
        class_offsets = class_offsets
    else:
        # TODO: This needs to be adapted to work with svm
        raise ValueError('SVM filters from landmarks not implemented yet!')
        # n_patches, n_classes = class_patches.shape[:2]
        # patch_shape = class_patches.shape[-3:]
        # if n_filters is None:
        #     n_filters = np.minimum(n_classes, np.prod(patch_shape)) - 1
        # # reshape patches
        # class_patches = np.rollaxis(class_patches, 1, 0)
        # class_patches = class_patches.reshape((n_patches * n_classes, -1))
        # # generate class offsets
        # class_offsets = [n_patches * j for j in range(1, n_classes)]

    # learn svm filters
    clf = svm.LinearSVC(class_weight='auto')
    clf.fit(class_patches, class_offsets)
    svm_filters = clf.coef_
    return svm_filters.reshape((-1,) + patch_shape)[..., ::-1, ::-1]


class DiscriminativeLDCN(LearnableLDCN):
    r"""
    Discriminative Linear Deep Convolutional Network Class
    """
    def __init__(self, learn_filters=learn_mccf_filters, n_layers=3,
                 architecture=3, normalize_patches=centralize,
                 normalize_filters=None, patch_shape=(31, 31),
                 mode='same', boundary='constant'):
        super(DiscriminativeLDCN, self).__init__(architecture=architecture,
                                                 patch_shape=patch_shape,
                                                 mode=mode, boundary=boundary)
        self._learn_filters = learn_filters
        self._n_layers = n_layers
        self.normalize_patches = normalize_patches
        self.normalize_filters = normalize_filters
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

        n_filters = [None] * self._n_layers
        if 'n_filters' in kwargs.keys():
            n_filters, self._n_layers = _parse_n_filters(
                kwargs.pop('n_filters'), self._n_layers)

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
            if self.normalize_patches:
                patches = _normalize_images(patches,
                                            norm_func=self.normalize_patches)
            # learn level filters
            fs = self._learn_filters(patches, patch_shape=self.patch_shape,
                                     verbose=verbose, level_str=level_str,
                                     n_filters=n_filters[l], **kwargs)
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
            if l < self._n_layers-1:
                # if not last layer, compute responses
                images = self.compute_filters_responses(
                    images, fs, norm_func=self.normalize_filters,
                    mode=self.mode, boundary=self.boundary, verbose=verbose,
                    string=level_str)
            if verbose:
                print_dynamic('{}Done!\n'.format(level_str))
        self._filters = filters
        self._n_filters = _parse_filters(filters)

    def learn_network_from_class_labels(self, images, class_labels, group=None,
                                        verbose=False, **kwargs):
        if verbose:
            print_dynamic('- Learning network\n')

        n_filters = [None] * self._n_layers
        if 'n_filters' in kwargs.keys():
            n_filters, self._n_layers = _parse_n_filters(
                kwargs.pop('n_filters'), self._n_layers)
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

            class_patches = []
            for label in class_labels:
                # extract patches
                patches = _extract_patches(
                    images, extract=_extract_patches_from_landmarks,
                    patch_shape=self.patch_shape, group=group, label=label,
                    as_single_array=True, flatten=True, verbose=verbose,
                    level_str=level_str)
                # normalize patches
                if self.normalize_patches:
                    patches = _normalize_images(
                        patches, norm_func=self.normalize_patches)
                class_patches.append(patches)
            if verbose:
                string = level_str + 'Learning filters '
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(0, show_bar=True)))

            # learn level filters
            fs = self._learn_filters(class_patches,
                                     patch_shape=self.patch_shape,
                                     verbose=verbose, level_str=level_str,
                                     n_filters=n_filters[l], **kwargs)
            # delete patches
            del class_patches, patches
            # normalize filters
            if self.normalize_filters:
                fs = _normalize_images(fs, norm_func=self.normalize_filters)
            # save filters
            filters.append(fs)
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(1, show_bar=True)))
            if l < self._n_layers-1:
                # if not last layer, compute responses
                images = self.compute_filters_responses(
                    images, fs, norm_func=self.normalize_filters,
                    mode=self.mode, boundary=self.boundary, verbose=verbose,
                    string=level_str)
            if verbose:
                print_dynamic('{}Done!\n'.format(level_str))
        self._filters = filters
        self._n_filters = _parse_filters(filters)