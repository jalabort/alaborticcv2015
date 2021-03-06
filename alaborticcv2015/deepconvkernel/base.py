from __future__ import division
import abc
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import warnings
from menpo.math import pad, crop, fft_convolve2d, fft_convolve2d_sum
from menpo.image import Image
from menpo.feature import ndfeature, centralize
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.utils import (
    normalize_images, extract_patches, extract_patches_from_grid,
    extract_patches_from_landmarks, images_to_image, image_to_images)


def _parse_filters(filters):
    n_filters = []
    for fs in filters:
        n_filters.append(fs.shape[0])
    return n_filters


def _check_stride(stride):
    if isinstance(stride, np.int):
        return stride, stride
    elif len(stride) == 1:
        return stride[0], stride[0]
    elif len(stride) == 2:
        return stride
    else:
        raise ValueError('stide={}, must be and integer or tuple of '
                         'integers').format(stride)


def _check_layer(layer, n_layers):
    if layer is None:
        layer = n_layers
    elif 0 < layer >= n_layers:
        layer = n_layers
        warnings.warn('layer={} must be an integer between '
                      '0 and {}. Response will be computed using all {}'
                      'layers of the network.').format(layer, n_layers,
                                                       n_layers)
    return layer


def _compute_kernel1(filters, ext_shape=None):
    kernel = 1
    for fs in filters:
        if ext_shape is not None:
            fs = pad(fs, ext_shape)
        fft_fs = fft2(fs)
        kernel *= np.sum(fft_fs.conj() * fft_fs, axis=(0, 1))
    return np.real(kernel)


def _compute_kernel2(filters, ext_shape=None):
    if len(filters) > 1:
        prev_kernel = _compute_kernel2(filters[1:], ext_shape=ext_shape)
        fs = filters[0]
        if ext_shape is not None:
            fs = pad(fs, ext_shape=ext_shape)
        fft_fs = fft2(fs)
        kernel = (np.sum(fft_fs.conj(), axis=0) *
                  np.sum(prev_kernel, axis=0)[None] *
                  np.sum(fft_fs, axis=0))
    else:
        fs = filters[0]
        if ext_shape is not None:
            fs = pad(fs, ext_shape=ext_shape)
        fft_fs = fft2(fs)
        kernel = np.sum(fft_fs.conj(), axis=0) * np.sum(fft_fs, axis=0)
    return np.real(kernel)


def __compute_kernel3(filters, ext_shape=None):
    if len(filters) > 1:
        aux1, aux2 = __compute_kernel3(filters[1:], ext_shape=ext_shape)
        fs = filters[0]
        if ext_shape is not None:
            fs = pad(fs, ext_shape=ext_shape)
        fft_fs = fft2(fs)
        aux1 = np.sum(fft_fs[None] * aux1[:, :, None, ...], axis=1)
        aux2 = np.sum(fft_fs.conj()[None] * aux2[:, :, None, ...], axis=1)
    else:
        fs = filters[0]
        if ext_shape is not None:
            fs = pad(fs, ext_shape=ext_shape)
        aux1 = fft2(fs)
        aux2 = aux1.conj()
    return aux1, aux2


def _compute_kernel3(filters, ext_shape=None):
    aux1, aux2 = __compute_kernel3(filters, ext_shape=ext_shape)
    return np.real(np.sum(aux1 * aux2, axis=0))


def _compute_filters_responses1(images, filters, norm_func=centralize,
                                mode='same', boundary='symmetric',
                                verbose=False, string=''):
    if verbose:
        string += 'Computing Filter Responses '
    n_images = len(images)
    responses = []
    for j, i in enumerate(images):
        if verbose:
            print_dynamic('{}{}'.format(
                string, progress_bar_str(j/n_images, show_bar=True)))
        if norm_func:
            i = norm_func(i)
        r = fft_convolve2d(i, filters, mode=mode, boundary=boundary)
        responses += image_to_images(r)
    if verbose:
        print_dynamic('{}{}'.format(
            string, progress_bar_str(1, show_bar=True)))
    return responses


def _compute_filters_responses2(images, filters, norm_func=centralize,
                                mode='same', boundary='symmetric',
                                verbose=False, string=''):
    if verbose:
        string += 'Computing Filter Responses '
    n_images = len(images)
    responses = []
    for j, i in enumerate(images):
        if verbose:
            print_dynamic('{}{}'.format(
                string, progress_bar_str(j/n_images, show_bar=True)))
        if norm_func:
            i = norm_func(i)
        r = fft_convolve2d_sum(i, filters, mode=mode, boundary=boundary,
                               axis=0)
        responses.append(r)
    if verbose:
        print_dynamic('{}{}'.format(
            string, progress_bar_str(1, show_bar=True)))
    return responses


def _compute_filters_responses3(images, filters, norm_func=centralize,
                                mode='same', boundary='symmetric',
                                verbose=False, string=''):
    if verbose:
        string += 'Computing Filter Responses '
    n_images = len(images)
    responses = []
    for j, i in enumerate(images):
        if verbose:
            print_dynamic('{}{}'.format(
                string, progress_bar_str(j/n_images, show_bar=True)))
        if norm_func:
            i = norm_func(i)
        r = fft_convolve2d_sum(i, filters, mode=mode, boundary=boundary,
                               axis=1)
        responses.append(r)
    if verbose:
        print_dynamic('{}{}'.format(
            string, progress_bar_str(1, show_bar=True)))
    return responses


def _network_response1(x, filters, norm_func=centralize, hidden_mode='same',
                       visible_mode='valid', boundary='symmetric'):
    limit = len(filters) - 1
    xs = [x]
    for j, fs in enumerate(filters):
        if norm_func:
            xs = normalize_images(xs, norm_func=norm_func)
        nxs = []
        for x in xs:
            if j < limit:
                x = fft_convolve2d(x, fs, mode=hidden_mode, boundary=boundary)
            else:
                x = fft_convolve2d(x, fs, mode=visible_mode, boundary=boundary)
            nxs += image_to_images(x)
        xs = nxs
    if isinstance(x, Image):
        return images_to_image(xs)
    else:
        return np.asarray(xs).reshape((-1, x.shape[-2:]))


def _network_response2(x, filters, norm_func=centralize, hidden_mode='same',
                       visible_mode='valid', boundary='symmetric'):
    limit = len(filters) - 1
    for j, fs in enumerate(filters):
        if norm_func:
            x = norm_func(x)
        if j < limit:
            x = fft_convolve2d_sum(x, fs, mode=hidden_mode,
                                   boundary=boundary)
        else:
            x = fft_convolve2d_sum(x, fs, mode=visible_mode,
                                   boundary=boundary)
    return x


def _network_response3(x, filters, norm_func=centralize, hidden_mode='same',
                       visible_mode='valid', boundary='symmetric'):
    limit = len(filters) - 1
    for j, fs in enumerate(filters):
        if norm_func:
            x = norm_func(x)
        if j < limit:
            x = fft_convolve2d_sum(x, fs, mode=hidden_mode,
                                   boundary=boundary, axis=1)
        else:
            x = fft_convolve2d_sum(x, fs, mode=visible_mode,
                                   boundary=boundary, axis=1)
    return x


@ndfeature
def _kernel_response(x, compute_kernel, filters_shape, layer=None,
                     norm_func=centralize, mode='valid',
                     boundary='symmetric'):
    if norm_func:
        x = norm_func(x)
    # extended shape
    x_shape = np.asarray(x.shape[-2:])
    f_shape = np.asarray(filters_shape)
    ext_shape = x_shape + f_shape - 1

    # extend image
    ext_x = pad(x, ext_shape, boundary=boundary)
    # compute extended image fft
    fft_ext_image = fft2(ext_x)

    # compute deep convolution kernel
    fft_ext_kernel = compute_kernel(layer=layer, ext_shape=ext_shape)

    # compute kernel extended response in Fourier domain
    fft_ext_c = fft_ext_kernel**0.5 * fft_ext_image

    # compute ifft of extended response
    ext_c = np.real(ifft2(fft_ext_c))

    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, x_shape)
    elif mode is 'valid':
        return crop(ext_c, x_shape - f_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))


class LinDeepConvNet(object):
    r"""
    Linear Deep Convolutional Network Interface
    """
    def __init__(self, architecture=3, norm_func=centralize):
        if architecture == 1:
            self._network_response = _network_response1
            self.__compute_kernel = _compute_kernel1
        elif architecture == 2:
            self._network_response = _network_response2
            self.__compute_kernel = _compute_kernel2
        elif architecture == 3:
            self._network_response = _network_response3
            self.__compute_kernel = _compute_kernel3
        else:
            raise ValueError('architecture={} must be an integer between 1 '
                             'and 3.').format(architecture)
        self.norm_func = norm_func

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
            n_filters.append(np.sum(n))
        return n_filters

    @property
    def filters_shape(self):
        return self._filters[0].shape[-2:]

    def filters_spatial(self):
        filters = []
        for fs in self._filters:
            level_filters = []
            for f in fs:
                level_filters.append(Image(f))
            filters.append(level_filters)
        return filters

    def filters_frequency(self, ext_shape=None):
        filters = []
        for fs in self._filters:
            level_filters = []
            for f in fs:
                if ext_shape is not None:
                    f = pad(f, ext_shape=ext_shape)
                level_filters.append(Image(np.abs(fftshift(fft2(f),
                                                           axes=(-2, -1)))))
            filters.append(level_filters)
        return filters

    def kernels_spatial(self, ext_shape=None):
        kernels = []
        for layer in range(self.n_layers):
            fft_kernel = self._compute_kernel(layer=layer, ext_shape=ext_shape)
            kernels.append(Image(np.real(fftshift(ifft2(fft_kernel),
                                         axes=(-2, -1)))))
        return kernels

    def kernels_frequency(self, ext_shape=None):
        kernels = []
        for layer in range(self.n_layers):
            fft_kernel = self._compute_kernel(layer=layer, ext_shape=ext_shape)
            kernels.append(Image(fftshift(fft_kernel, axes=(-2, -1))))
        return kernels

    def _compute_kernel(self, layer=None, ext_shape=None):
        layer = _check_layer(layer, self.n_layers)
        return self.__compute_kernel(self._filters[:layer+1],
                                     ext_shape=ext_shape)

    def network_response(self, image, layer=None, hidden_mode='same',
                         visible_mode='valid', boundary='symmetric'):
        layer = _check_layer(layer, self.n_layers)
        return self._network_response(image, self._filters[:layer+1],
                                      norm_func=self.norm_func,
                                      hidden_mode=hidden_mode,
                                      visible_mode=visible_mode,
                                      boundary=boundary)

    def kernel_response(self, x, layer=None, mode='valid',
                        boundary='symmetric'):
        return _kernel_response(x, self._compute_kernel, self.filters_shape,
                                layer=layer, norm_func=self.norm_func,
                                mode=mode, boundary=boundary)


class LearnableLDCN(LinDeepConvNet):
    r"""
    Learnable Linear Deep Convolutional Network Interface
    """
    def __init__(self, architecture=3, norm_func=centralize,
                 patch_shape=(7, 7), mode='same', boundary='constant'):
        super(LearnableLDCN, self).__init__(architecture=architecture,
                                            norm_func=norm_func)
        if architecture == 1:
            self.compute_filters_responses = _compute_filters_responses1
        elif architecture == 2:
            self.compute_filters_responses = _compute_filters_responses2
        elif architecture == 3:
            self.compute_filters_responses = _compute_filters_responses3
        else:
            raise ValueError('architecture={} must be an integer between 1 '
                             'and 3.').format(architecture)
        self.patch_shape = patch_shape
        self.mode = mode
        self.boundary = boundary

    def learn_network_from_grid(self, images, stride=4, verbose=False,
                                **kwargs):
        stride = _check_stride(stride)

        def extract_patches_func(imgs, flatten=True, string=''):
            return extract_patches(imgs, extract=extract_patches_from_grid,
                                   patch_shape=self.patch_shape,
                                   stride=stride, as_single_array=True,
                                   flatten=flatten, verbose=verbose,
                                   string=string)
        self._learn_network(images, extract_patches_func, verbose=verbose,
                            **kwargs)

    def learn_network_from_landmarks(self, images, group=None, label=None,
                                     verbose=False, **kwargs):
        def extract_patches_func(imgs, flatten=True, string=''):
            return extract_patches(imgs,
                                   extract=extract_patches_from_landmarks,
                                   patch_shape=self.patch_shape, group=group,
                                   label=label, as_single_array=True,
                                   flatten=flatten, verbose=verbose,
                                   string=string)
        self._learn_network(images, extract_patches_func, verbose=verbose,
                            **kwargs)

    @abc.abstractmethod
    def _learn_network(self, images, extract_patches_func, verbose=False,
                       **kwargs):
        pass