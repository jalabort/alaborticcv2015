from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import warnings
from menpo.image import Image
from menpo.feature import ndfeature
from alaborticcv2015.utils import (
    pad, crop, fft_convolve2d_sum)


def _parse_filters(filters):
    n_filters = []
    for fs in filters:
        n_filters.append(fs.shape[0])
    return n_filters


def _parse_filters_options(learn_filters, n_filters, n_layers):
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
            return _parse_filters_options(learn_filters[0], n_filters, n_layers)
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
                        lfs, nfs = _parse_filters_options(lfs, nfs, 1)
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


def _check_stride(stride):
    if len(stride) == 1:
        return stride, stride
    if len(stride) == 2:
        return stride


def _check_layer(layer, n_layers):
    if layer is None:
        layer = n_layers
    elif 0 < layer >= n_layers:
        layer = n_layers
        warnings.warn('layer={} must be an integer between '
                      '0 and {}. Response will be computed using all {}'
                      'layers of the network.') .format(layer,
                                                        n_layers,
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


@ndfeature
def _network_response(x, filters, hidden_mode='same', visible_mode='valid',
                      boundary='symmetric'):
    limit = len(filters) - 1
    for j, fs in enumerate(filters):
        if j < limit:
            x = fft_convolve2d_sum(x, fs, mode=hidden_mode,
                                   boundary=boundary, axis=1)
        else:
            x = fft_convolve2d_sum(x, fs, mode=visible_mode,
                                   boundary=boundary, axis=1)
    return x


@ndfeature
def _kernel_response(x, compute_kernel, filters_shape, layer=None,
                     mode='valid', boundary='symmetric'):
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
        return _compute_kernel3(self._filters[:layer+1], ext_shape=ext_shape)

    def network_response(self, image, layer=None, hidden_mode='same',
                         visible_mode='same', boundary='constant'):
        layer = _check_layer(layer, self.n_layers)
        return _network_response(image, self._filters[:layer+1],
                                 hidden_mode=hidden_mode,
                                 visible_mode=visible_mode, boundary=boundary)

    def kernel_response(self, x, layer=None, mode='same', boundary='constant'):
        return _kernel_response(x, self._compute_kernel, self.filters_shape,
                                layer=layer, mode=mode, boundary=boundary)

