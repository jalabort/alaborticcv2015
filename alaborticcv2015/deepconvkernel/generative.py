from __future__ import division
import warnings
from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math.decomposition.ica import _batch_ica, negentropy_logcosh
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.utils import normalize_patches, centralize
from .base import LearnableLDCN, compute_filters_responses


def learn_pca_filters(patches, n_filters=8, normalize=centralize, dtype=None):
    r"""
    Learn PCA convolution _filters
    """
    if normalize:
        # normalize patches if required
        for j, p in enumerate(patches):
            patches[j] = normalize(p, dtype=dtype)
    # learn pca model
    pca = PCAModel(patches)
    # set active number of components
    pca.n_active_components = n_filters
    # obtain and return _filters
    return [pca.template_instance.from_vector(pc) for pc in pca.components]


def learn_ica_filters(patches, n_filters=8, normalize=centralize, dtype=None,
                      algorithm=_batch_ica, negentropy=negentropy_logcosh,
                      max_iters=500):
    r"""
    Learn ICA convolution _filters
    """
    if normalize:
        # normalize patches if required
        for j, p in enumerate(patches):
            patches[j] = normalize(p, dtype=dtype)
    # learn ica model
    ica = ICAModel(patches, algorithm=algorithm, negentropy_approx=negentropy,
                   n_components=n_filters, max_iters=max_iters)
    # obtain and return _filters
    return [ica.template_instance.from_vector(ic) for ic in ica.components]


def learn_nmf_filters(patches, n_filters=8, max_iters=500, verbose=False,
                      **kwargs):
    r"""
    Learn NMF convolution _filters
    """
    # learn nmf model
    nmf = NMFModel(patches, n_components=n_filters, max_iters=max_iters,
                   verbose=verbose, **kwargs)
    # obtain and return _filters
    return [nmf.template_instance.from_vector(nf) for nf in nmf.components]


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
                 n_layers=3, patch_shape=(7, 7), norm_func=centralize,
                 mode='same', boundary='constant'):

        self._learn_filters, self._n_filters = _parse_params(
            learn_filters, n_filters, n_layers)

        self.patch_shape = patch_shape
        self.norm_func = norm_func
        self.mode = mode
        self.boundary = boundary

    def _learn_network(self, images, extract_patches_func, verbose=False,
                       **kwargs):
        if verbose:
            string = '- Learning network'

        # # convert images to the appropriate type
        # convert_images_to_dtype_inplace(images, dtype=self.dtype)

        filters = []
        for j, (lfs, nfs) in enumerate(zip(self._learn_filters,
                                           self._n_filters)):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_layers, show_bar=True)))

            # extract patches
            patches = extract_patches_func(images)
            # normalize patches
            patches = normalize_patches(patches, norm_func=self.norm_func)
            # learn level _filters
            fs = []
            for (lf, nf) in zip(lfs, nfs):
                fs.append(lf(patches, nf, **kwargs))
            # delete patches
            del patches
            # normalize filters
            fs = normalize_patches(fs, norm_func=self.norm_func)
            # save filters
            filters.append(fs)
            if j != self.n_layers:
                # compute responses if not last layer
                images = compute_filters_responses(images, fs,
                                                   norm_func=self.norm_func)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))
