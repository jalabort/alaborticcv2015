from __future__ import division
import numpy as np
from menpo.math import mccf, lda
from menpo.feature import centralize
from menpo.visualize import print_dynamic, progress_bar_str
from alaborticcv2015.utils import normalize_images
from .base import LearnableLDCN, compute_filters_responses


def learn_mccf_filters(patches):
    r"""
    Learn MCCF convolution filters
    """
    n_patches = patches.shape[0]
    patch_shape = patches.shape[-3:]
    patches = patches.reshape((n_patches, -1))
    # learn pca filters
    lda_filters = mccf(patches, inplace=True)[0]
    return lda_filters.reshape((-1,) + patch_shape)


def learn_lda_filters(patches, n_filters=8):
    r"""
    Learn LDA convolution filters
    """
    n_patches = patches.shape[0]
    patch_shape = patches.shape[-3:]
    patches = patches.reshape((n_patches, -1))
    # learn lda filters
    lda_filters = lda(patches, inplace=True)[0]
    lda_filters = lda_filters[:n_filters, :]
    return lda_filters.reshape((-1,) + patch_shape)


class DiscriminativeLDCN():

    def __init__(self, learn_filters=learn_lda_filters, n_levels=3,
                 n_filters=8, patch_size=(7, 7), mean_centre=True,
                 padding='constant'):
        self._learn_filters = learn_filters
        self.n_levels = n_levels
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.padding = padding

    def learn_network(self, images, group=None, label=None,
                      verbose=False, **kwargs):
        if verbose:
            string = '- Learning network'
        # extract centres
        centres = [i.landmarks[group][label] for i in images]
        # initialize level_image and list of _filters
        level_images = images
        self.filters = []
        for j in range(self.n_levels):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_levels, show_bar=True)))
            # extract level patches
            level_patches, level_classes = self._extract_patches(level_images,
                                                                 centres)
            # learn level _filters
            level_filters = self._learn_filters(level_patches, level_classes,
                                                self.n_filters, **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            # save level _filters
            self.filters.append(level_filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

    def _extract_patches(self, images, centres):
        patches = []
        classes = []
        for i, c in zip(images, centres):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            for j, p in enumerate(i_patches):
                patches.append(p)
                classes.append(j)
        return patches, classes

    def _compute_filter_responses(self, images, filters):
        # compute  and return list of responses
        return [self._apply_filters(i, filters) for i in images]

    def _apply_filters(self, image, filters):
        # define extended shape
        ext_h = image.shape[0] + 2 * filters[0].shape[0]
        ext_w = image.shape[1] + 2 * filters[0].shape[1]
        ext_shape = (filters[0].n_channels, ext_h, ext_w)
        # define response extended shape
        r_ext_shape = (len(filters), image.shape[0], image.shape[1])

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image)

        # initialize response
        response = np.zeros(r_ext_shape)
        for j, f in enumerate(filters):

            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)

            # compute extended filter fft
            fft_ext_f = fft2(ext_f)

            # compute filter response
            fft_ext_r = np.sum(fft_ext_f * fft_ext_image, axis=0)[None]
            # compute inverse fft of the filter response
            ext_r = fftshift(np.real(ifft2(fft_ext_r)), axes=(-2, -1))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # fill out overall response
            response[j] = r

        # transform response to Image and return it
        return Image(response)

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = self._learn_kernel(self.filters[:level], ext_shape=ext_shape)
        return Image(fftshift(kernel, axes=(-2, -1)))

    def _learn_kernel(self, filters, ext_shape=None):
        _, ext_h, ext_w = ext_shape
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                kernel += np.real(fft_f.conj() * fft_f)

        return kernel

    def compute_network_response(self, image, level=None):
        for level_filters in self.filters[:level]:
            image = self._apply_filters(image, level_filters)
        return image

    def compute_kernel_response(self, image, level=None):
        # obtain number of channels
        n_ch, h, w = self.filters[0][0].pixels.shape

        # obtain extended shape
        ext_h = image.shape[0] + 2 * h
        ext_w = image.shape[1] + 2 * w
        ext_shape = (n_ch, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image)

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(ifft2(fft_ext_r))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)
