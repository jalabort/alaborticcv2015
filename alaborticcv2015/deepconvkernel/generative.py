from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

import warnings

from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math.decomposition.ica import _batch_ica, negentropy_logcosh
from menpo.image import Image
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit.builder import normalization_wrt_reference_shape

from alaborticcv2015.utils import (
    pad, crop, fft_convolve2d, fft_convolve2d_sum,
    multiconvsum,
    convert_images_to_dtype_inplace, extract_patches,
    extract_patches_from_grid, extract_patches_from_landmarks,
    centralize)


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


def parse_filter_options(learn_filters, n_filters, n_layers):
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
            return parse_filter_options(learn_filters[0], n_filters, n_layers)
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
                        lfs, nfs = parse_filter_options(lfs, nfs, 1)
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


def parse_stride(stride):
    if len(stride) == 1:
        return stride, stride
    if len(stride) == 2:
        return stride


class GenerativeNetwork():

    def __init__(self, learn_filters=learn_pca_filters, n_filters=8,
                 n_layers=3, patch_size=(7, 7), normalize=centralize,
                 mode='same', boundary='constant', dtype=np.float32):

        self._learn_filters, self._n_filters = parse_filter_options(
            learn_filters, n_filters, n_layers)

        self.patch_size = patch_size
        self.normalize = normalize
        self.mode = mode
        self.boundary = boundary
        self.dtype = dtype

        self._filters = None
        self._extract_patches = None

    @property
    def n_layers(self):
        return len(self._n_filters)

    @property
    def n_filters(self):
        n_filters = 0
        for n in self._n_filters:
            n_filters += np.sum(n)
        return n_filters

    @property
    def n_filters_layer(self):
        n_filters = []
        for n in self._n_filters:
            n_filters += [np.sum(n)]
        return n_filters

    def filters_spatial(self):
        filters = []
        for fs in self._filters:
            for j, f in enumerate(fs):
                fs[j] = Image(f)
            filters.append(fs)
        return filters

    def filters_frequency(self):
        filters = []
        for fs in self._filters:
            for j, f in enumerate(fs):
                fs[j] = Image(np.abs(ifftshift(fft2(f))))
            filters.append(fs)
        return filters

    def learn_network_from_landmarks(self, images, group=None, label=None,
                                     verbose=False, **kwargs):

        def extract_patches_function(imgs, patch_shape=(7, 7), dtype=None):
            return extract_patches(
                imgs, extract=extract_patches_from_landmarks,
                patch_shape=patch_shape, dtype=dtype, group=group, label=label)

        self._learn_network(images, extract_patches_function, verbose=verbose,
                            **kwargs)

    def learn_network_from_grid(self, images, stride=4, verbose=False,
                                **kwargs):

        stride = parse_stride(stride)

        def extract_patches_function(imgs, patch_shape=(7, 7), dtype=None):
            return extract_patches(imgs, extract=extract_patches_from_grid,
                                   patch_shape=patch_shape, dtype=dtype,
                                   stride=stride)

        self._learn_network(images, extract_patches_function, verbose=verbose,
                            **kwargs)

    def _learn_network(self, images, extract_patches_function, verbose=False,
                       **kwargs):

        if verbose:
            string = '- Learning network'

        # convert images to the appropriate type
        convert_images_to_dtype_inplace(images, dtype=self.dtype)

        # initialize list of _filters
        self._filters = []

        for j, (lfs, nfs) in enumerate(zip(self._learn_filters,
                                           self._n_filters)):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_layers, show_bar=True)))

            # extract level patches
            patches = extract_patches_function(images,
                                               patch_shape=self.patch_size,
                                               dtype=self.dtype)

            # learn level _filters
            level_filters = []
            for (lf, nf) in zip(lfs, nfs):
                level_filters += lf(patches, nf, **kwargs)

            # delete patches
            del patches

            # compute level responses lists
            images = self._compute_filter_responses(images, level_filters)
            # save level _filters
            self._filters.append(level_filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

    def _compute_filter_responses(self, images, filters):
        return [multiconvsum(i, filters, mode=self.mode,
                             boundary=self.boundary)
                for i in images]

    def _compute_filter_responses(self, images, filters):
        return [fft_convolve2d_sum(i, filters, mode=self.mode,
                                   boundary=self.boundary)
                for i in images]

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = self._learn_kernel(self._filters[:level],
                                    ext_shape=ext_shape)
        return Image(np.real(kernel))

    def _learn_kernel(self, filters, ext_shape=None):
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                fft_ext_f = fft2(f.pixels)
                kernel += fft_ext_f.conj() * prev_kernel[j] * fft_ext_f
        else:
            kernel = 0
            for f in filters[0]:
                fft_ext_f = fft2(f.pixels)
                kernel += fft_ext_f.conj() * fft_ext_f

        return kernel

    def compute_network_response(self, image, level=None):
        for j, fs in enumerate(self._filters[:level]):
            if j < level:
                image = multiconvsum(image, fs, mode=self.mode,
                                     boundary=self.boundary)
            else:
                image = multiconvsum(image, fs, mode='same',
                                     boundary=self.boundary)
        return image

    def compute_kernel_response(self, image, level=None):
        # obtain number of channels
        n_ch, h, w = self._filters[0][0].pixels.shape

        # obtain extended shape
        ext_h = image.shape[0] + h - 1
        ext_w = image.shape[1] + w - 1
        ext_shape = (ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.boundary)
        # compute extended image fft
        fft_ext_image = fft2(ext_image)

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fft_ext_kernel.pixels**0.5 * fft_ext_image

        # compute inverse fft of deep response
        r = np.real(ifft2(fft_ext_r))
        r = crop(r, image.shape)

        return Image(r)


# class GenerativeDCKArch2():
#
#     def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
#                  n_filters=8, patch_size=(7, 7), mean_centre=True,
#                  padding='constant'):
#         self._learn_filters = learn_filters
#         self.n_levels = n_levels
#         self.n_filters = n_filters
#         self.patch_size = patch_size
#         self.mean_centre = mean_centre
#         self.padding = padding
#
#     def learn_network(self, images, group=None, label=None,
#                       verbose=False, **kwargs):
#         if verbose:
#             string = '- Learning network'
#         # extract centres
#         centres = [i.landmarks[group][label] for i in images]
#         # initialize level_image and list of _filters
#         level_images = images
#         self._filters = []
#         for j in xrange(self.n_levels):
#             if verbose:
#                 print_dynamic('{}: {}'.format(
#                     string, progress_bar_str(j/self.n_levels, show_bar=True)))
#             # extract level patches
#             level_patches = self._extract_patches(level_images, centres)
#             # learn level _filters
#             level_filters = self._learn_filters(level_patches, self.n_filters,
#                                                 **kwargs)
#             # compute level responses lists
#             level_images = self.compute_filter_responses(level_images,
#                                                           level_filters)
#             # save level _filters
#             self._filters.append(level_filters)
#
#         if verbose:
#             print_dynamic('{}: Done!\n'.format(string))
#
#     def _extract_patches(self, images, centres):
#         patches = []
#         for i, c in zip(images, centres):
#             i_patches = i.extract_patches(c, patch_size=self.patch_size)
#             patches.append(i_patches)
#         return [i for i_patches in patches for i in i_patches]
#
#     def compute_filter_responses(self, images, _filters):
#         # compute responses lists
#         images = [self._apply_filters(i, _filters) for i in images]
#         # flatten out and return list of responses
#         return [i for img in images for i in img]
#
#     def _apply_filters(self, image, _filters):
#         # define extended shape
#         ext_h = image.shape[0] + _filters[0].shape[0] - 1
#         ext_w = image.shape[1] + _filters[0].shape[1] - 1
#         ext_shape = (_filters[0].n_channels, ext_h, ext_w)
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))
#
#         # initialize list of responses
#         responses = []
#         for j, f in enumerate(_filters):
#             # extend filter
#             ext_f = pad(f.pixels, ext_shape, mode=self.padding)
#             # compute extended filter fft
#             fft_ext_f = fftshift(fft2(ext_f), axes=(-2, -1))
#
#             # compute filter response
#             fft_ext_r = fft_ext_f * fft_ext_image
#             # compute inverse fft of the filter response
#             ext_r = fftshift(np.real(ifft2(ifftshift(fft_ext_r,
#                                                      axes=(-2, -1)))),
#                              axes=(-2, -1))
#
#             # crop response to original image size
#             r = unpad(ext_r, image.shape)
#             # transform response to Image
#             responses.append(Image(r))
#
#         # return list of responses
#         return responses
#
#     def learn_kernel(self, level=None, ext_shape=None):
#         kernel = 1
#         for level_filters in self._filters[:level]:
#             k = 0
#             for f in level_filters:
#                 if ext_shape:
#                     f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
#                 else:
#                     f_pixels = f.pixels
#                 fft_f = fftshift(fft2(f_pixels), axes=(-2, -1))
#                 k += fft_f.conj() * fft_f
#             kernel *= k
#         return Image(kernel)
#
#     def compute_network_response(self, image, level=None):
#         images = [image]
#         for level_filters in self._filters[:level]:
#             images = self.compute_filter_responses(images, level_filters)
#         return self._list_to_image(images)
#
#     @classmethod
#     def _list_to_image(cls, images):
#         img = images[0]
#         n_ch, h, w = img.pixels.shape
#         pixels = np.zeros((len(images) * n_ch, h, w))
#         for j, i in enumerate(images):
#             ind = j * n_ch
#             pixels[ind:ind+n_ch] = i.pixels
#         return Image(pixels)
#
#     def compute_kernel_response(self, image, level=None):
#         # obtain number of channels
#         n_ch, h, w = self._filters[0][0].pixels.shape
#
#         # obtain extended shape
#         ext_h = image.shape[0] + h - 1
#         ext_w = image.shape[1] + w - 1
#         ext_shape = (n_ch, ext_h, ext_w)
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))
#
#         # compute deep convolutional kernel
#         fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)
#
#         # compute deep response by convolving image with sqrt of deep kernel
#         fft_ext_r = fft_ext_kernel.pixels**0.5 * fft_ext_image
#
#         # compute inverse fft of deep response
#         ext_r = np.real(ifft2(ifftshift(fft_ext_r, axes=(-2, -1))))
#
#         # crop deep response to original image size and return it
#         r = unpad(ext_r, image.shape)
#         return Image(r)
#
#
# class GenerativeDCKArch3():
#
#     def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
#                  n_filters=8, patch_size=(7, 7), mean_centre=True,
#                  padding='constant'):
#         self._learn_filters = learn_filters
#         self.n_levels = n_levels
#         self.n_filters = n_filters
#         self.patch_size = patch_size
#         self.mean_centre = mean_centre
#         self.padding = padding
#
#     def learn_network(self, images, group=None, label=None,
#                       verbose=False, **kwargs):
#         if verbose:
#             string = '- Learning network'
#         # extract centres
#         centres = [i.landmarks[group][label] for i in images]
#         # initialize level_image and list of _filters
#         level_images = images
#         self._filters = []
#         for j in range(self.n_levels):
#             if verbose:
#                 print_dynamic('{}: {}'.format(
#                     string, progress_bar_str(j/self.n_levels, show_bar=True)))
#             # extract level patches
#             level_patches = self._extract_patches(level_images, centres)
#             # learn level _filters
#             level_filters = self._learn_filters(level_patches, self.n_filters,
#                                                 **kwargs)
#             # compute level responses lists
#             level_images = self.compute_filter_responses(level_images,
#                                                           level_filters)
#             # save level _filters
#             self._filters.append(level_filters)
#
#         if verbose:
#             print_dynamic('{}: Done!\n'.format(string))
#
#     def _extract_patches(self, images, centres):
#         patches = []
#         for i, c in zip(images, centres):
#             i_patches = i.extract_patches(c, patch_size=self.patch_size)
#             patches.append(i_patches)
#         return [i for i_patches in patches for i in i_patches]
#
#     def compute_filter_responses(self, images, _filters):
#         # compute  and return list of responses
#         return [self._apply_filters(i, _filters) for i in images]
#
#     def _apply_filters(self, image, _filters):
#         # define extended shape
#         ext_h = image.shape[0] + _filters[0].shape[0] - 1
#         ext_w = image.shape[1] + _filters[0].shape[1] - 1
#         N = np.sqrt(ext_h * ext_w)
#         ext_shape = (_filters[0].n_channels, ext_h, ext_w)
#         # define response extended shape
#         r_ext_shape = (image.n_channels, image.shape[0], image.shape[1])
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fft2(ext_image)
#
#         # initialize response
#         response = 0
#         for f in _filters:
#             n_ch = f.shape[0]
#
#             # extend filter
#             ext_f = pad(f.pixels, ext_shape, mode=self.padding)
#
#             # compute extended filter fft
#             fft_ext_f = fft2(ext_f)
#             # compute filter response
#             fft_ext_r = fft_ext_f * fft_ext_image
#             # compute inverse fft of the filter response
#             ext_r = fftshift(np.real(ifft2(fft_ext_r)), axes=(-2, -1))
#
#             # crop response to original image size
#             r = unpad(ext_r, image.shape)
#             # fill out overall response
#             response += r
#
#         # transform response to Image and return it
#         return Image(response)
#
#     def learn_kernel(self, level=None, ext_shape=None):
#         kernel = self._learn_kernel(self._filters[:level], ext_shape=ext_shape)
#         return Image(fftshift(kernel, axes=(-2, -1)))
#
#     def _learn_kernel(self, _filters, ext_shape=None):
#         _, ext_h, ext_w = ext_shape
#         N = ext_h * ext_w
#         if len(_filters) > 1:
#             prev_kernel = self._learn_kernel(_filters[1:], ext_shape=ext_shape)
#             kernel = 0
#             for j, f in enumerate(_filters[0]):
#                 n_ch = f.shape[0]
#                 if ext_shape:
#                     f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
#                 else:
#                     f_pixels = f.pixels
#                 fft_f = fft2(f_pixels)
#                 kernel += fft_f * prev_kernel[j] * fft_f
#         else:
#             kernel = 0
#             for f in _filters[0]:
#                 n_ch = f.shape[0]
#                 if ext_shape:
#                     f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
#                 else:
#                     f_pixels = f.pixels
#                 fft_f = fft2(f_pixels)
#                 kernel += np.real(fft_f.conj() * fft_f)
#
#         return kernel
#
#     def compute_network_response(self, image, level=None):
#         for level_filters in self._filters[:level]:
#             image = self._apply_filters(image, level_filters)
#         return image
#
#     def compute_kernel_response(self, image, level=None):
#         # obtain number of channels
#         n_ch, h, w = self._filters[0][0].pixels.shape
#
#         # obtain extended shape
#         ext_h = image.shape[0] + h - 1
#         ext_w = image.shape[1] + w - 1
#         N = np.sqrt(ext_h * ext_w)
#         ext_shape = (n_ch, ext_h, ext_w)
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fft2(ext_image)
#
#         # compute deep convolutional kernel
#         fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)
#
#         # compute deep response by convolving image with sqrt of deep kernel
#         fft_ext_r = fftshift(
#             fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image
#
#         # compute inverse fft of deep response
#         ext_r = np.real(ifft2(fft_ext_r))
#
#         # crop deep response to original image size and return it
#         r = unpad(ext_r, image.shape)
#         return Image(r)
#
#
# class GenerativeDCKArch4():
#
#     def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
#                  n_filters=8, patch_size=(7, 7), mean_centre=True,
#                  padding='constant'):
#         self._learn_filters = learn_filters
#         self.n_levels = n_levels
#         self.n_filters = n_filters
#         self.patch_size = patch_size
#         self.mean_centre = mean_centre
#         self.padding = padding
#
#     def learn_network(self, images, group=None, label=None,
#                       verbose=False, **kwargs):
#         if verbose:
#             string = '- Learning network'
#         # extract centres
#         centres = [i.landmarks[group][label] for i in images]
#         # initialize level_image and list of _filters
#         level_images = images
#         self._filters = []
#         for j in xrange(self.n_levels):
#             if verbose:
#                 print_dynamic('{}: {}'.format(
#                     string, progress_bar_str(j/self.n_levels, show_bar=True)))
#
#             # extract level patches
#             level_patches = self._extract_patches(level_images, centres)
#
#             # learn level _filters
#             level_filters = self._learn_filters(level_patches, self.n_filters,
#                                                 **kwargs)
#             # save level _filters
#             self._filters.append(level_filters)
#
#             # compute level responses lists
#             level_images = self._compute_kernel_responses(images)
#
#         if verbose:
#             print_dynamic('{}: Done!\n'.format(string))
#
#     def _extract_patches(self, images, centres):
#         patches = []
#         for i, c in zip(images, centres):
#             i_patches = i.extract_patches(c, patch_size=self.patch_size)
#             patches.append(i_patches)
#         return [i for i_patches in patches for i in i_patches]
#
#     def _compute_kernel_responses(self, images):
#         # compute and return kernel responses
#         return [self.compute_kernel_response(i) for i in images]
#
#     def compute_kernel_response(self, image, level=None):
#         # obtain number of channels
#         n_ch, h, w = self._filters[0][0].pixels.shape
#
#         # obtain extended shape
#         ext_h = image.shape[0] + h - 1
#         ext_w = image.shape[1] + w - 1
#         ext_shape = (n_ch, ext_h, ext_w)
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fft2(ext_image)
#
#         # compute deep convolutional kernel
#         fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)
#
#         # compute deep response by convolving image with sqrt of deep kernel
#         fft_ext_r = fftshift(
#             fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image
#
#         # compute inverse fft of deep response
#         ext_r = np.real(ifft2(fft_ext_r))
#
#         # crop deep response to original image size and return it
#         r = unpad(ext_r, image.pixels.shape)
#         return Image(r)
#
#     def learn_kernel(self, level=None, ext_shape=None):
#         kernel = 1
#         for level_filters in self._filters[:level]:
#             k = 0
#             for f in level_filters:
#                 if ext_shape:
#                     f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
#                 else:
#                     f_pixels = f.pixels
#                 fft_f = fft2(f_pixels)
#                 k += np.real(fft_f.conj() * fft_f)
#             kernel *= k
#         return Image(fftshift(kernel, axes=(-2, -1)))
#
#     def compute_network_response(self, image, level=None):
#         images = [image]
#         for level_filters in self._filters[:level]:
#             images = self.compute_filter_responses(images, level_filters)
#         return self._list_to_image(images)
#
#     def compute_filter_responses(self, images, _filters):
#         # compute responses lists
#         images = [self._apply_filters(i, _filters) for i in images]
#         # flatten out and return list of responses
#         return [i for img in images for i in img]
#
#     def _apply_filters(self, image, _filters):
#         # define extended shape
#         ext_h = image.shape[0] + _filters[0].shape[0] - 1
#         ext_w = image.shape[1] + _filters[0].shape[1] - 1
#         ext_shape = (_filters[0].n_channels, ext_h, ext_w)
#
#         # extend image
#         ext_image = pad(image.pixels, ext_shape, mode=self.padding)
#         # compute extended image fft
#         fft_ext_image = fft2(ext_image)
#
#         # initialize list of responses
#         responses = []
#         for j, f in enumerate(_filters):
#             # extend filter
#             ext_f = pad(f.pixels, ext_shape, mode=self.padding)
#             # compute extended filter fft
#             fft_ext_f = fft2(ext_f)
#
#             # compute filter response
#             fft_ext_r = fft_ext_f * fft_ext_image
#             # compute inverse fft of the filter response
#             ext_r = np.real(fftshift(ifft2(fft_ext_r), axes=(-2, -1)))
#
#             # crop response to original image size
#             r = unpad(ext_r, image.shape)
#             # transform response to Image
#             responses.append(Image(r))
#
#         # return list of responses
#         return responses
#
#     @classmethod
#     def _list_to_image(cls, images):
#         img = images[0]
#         n_ch, h, w = img.pixels.shape
#         pixels = np.zeros((len(images) * n_ch, h, w))
#         for j, i in enumerate(images):
#             ind = j * n_ch
#             pixels[ind:ind+n_ch] = i.pixels
#         return Image(pixels)