from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from menpo.math import log_gabor
from menpo.image import Image

from alaborticcv2015.utils import pad, unpad


class LogGaborDCKArch1():

    def __init__(self, n_levels=3, patch_size=(7, 7), mean_centre=True,
                 n_scales=2, n_orientations=4, min_wavelength=2,
                 scaling_constant=2, center_sigma=0.65, d_phi_sigma=1.3,
                 padding='constant'):
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.n_scales = n_scales
        self.n_orientations = n_orientations 
        self.min_wavelength = min_wavelength
        self.scaling_constant = scaling_constant
        self.center_sigma = center_sigma
        self.d_phi_sigma = d_phi_sigma
        self.padding = padding

    def learn_network(self, images):
        # build log gabor
        _, _, _, gabor_filters = log_gabor(
            np.zeros(self.patch_size),
            num_scales=self.n_scales,
            num_orientations=self.n_orientations,
            min_wavelength=self.min_wavelength,
            scaling_constant=self.scaling_constant,
            center_sigma=self.center_sigma, 
            d_phi_sigma=self.d_phi_sigma)

        self.filters = []
        n_ch = images[0].n_channels
        template = Image(np.zeros((n_ch,) + self.patch_size))
        level_filters = []
        for gf in gabor_filters:
            # convert to spatial domain
            gf = np.real(fftshift(ifft2(gf[None])))
            # tile to have the same number of channels as the patches
            gf = np.tile(gf, (n_ch, 1, 1))
            # add to filter list
            level_filters.append(template.from_vector(gf.ravel()))
        self.filters.append(level_filters)

        self.n_filters = len(gabor_filters)
        template = Image(np.zeros((self.n_filters,) + self.patch_size))
        for j in range(self.n_levels - 1):
            level_filters = []
            for gf in gabor_filters:
                # convert to spatial domain
                gf = np.real(fftshift(ifft2(gf[None])))
                # tile to have the same number of channels as the patches
                gf = np.tile(gf, (self.n_filters, 1, 1))
                # add to filter list
                level_filters.append(template.from_vector(gf.ravel()))
            self.filters.append(level_filters)

    def _extract_patches(self, images, centres):
        patches = []
        for i, c in zip(images, centres):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

    def _compute_filter_responses(self, images, filters):
        # compute  and return list of responses
        return [self._apply_filters(i, filters) for i in images]

    def _apply_filters(self, image, filters):
        # define extended shape
        ext_h = image.shape[0] + filters[0].shape[0] - 1
        ext_w = image.shape[1] + filters[0].shape[1] - 1
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
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                kernel += np.sum(np.real(fft_f.conj() * prev_kernel * fft_f),
                                 axis=1)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                kernel += np.sum(np.real(fft_f.conj() * fft_f), axis=1)

        return kernel

    def compute_network_response(self, image, level=None):
        for level_filters in self.filters[:level]:
            image = self._apply_filters(image, level_filters)
        return image

    def compute_kernel_response(self, image, level=None):
        # obtain number of channels
        n_ch, h, w = self.filters[0][0].pixels.shape

        # obtain extended shape
        ext_h = image.shape[0] + h - 1
        ext_w = image.shape[1] + w - 1
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


class LogGaborDCKArch2():

    def __init__(self, n_levels=3, patch_size=(7, 7), mean_centre=True,
                 n_scales=2, n_orientations=4, min_wavelength=2,
                 scaling_constant=2, center_sigma=0.65, d_phi_sigma=1.3,
                 padding='constant'):
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wavelength = min_wavelength
        self.scaling_constant = scaling_constant
        self.center_sigma = center_sigma
        self.d_phi_sigma = d_phi_sigma
        self.padding = padding

    def learn_network(self, images):
        # build log gabor
        _, _, _, gabor_filters = log_gabor(
            np.zeros(self.patch_size),
            num_scales=self.n_scales,
            num_orientations=self.n_orientations,
            min_wavelength=self.min_wavelength,
            scaling_constant=self.scaling_constant,
            center_sigma=self.center_sigma,
            d_phi_sigma=self.d_phi_sigma)

        self.filters = []
        n_ch = images[0].n_channels
        template = Image(np.zeros((n_ch,) + self.patch_size))
        for j in range(self.n_levels - 1):
            level_filters = []
            for gf in gabor_filters:
                # convert to spatial domain
                gf = np.real(fftshift(ifft2(gf[None], axes=(-2, -1))))
                # tile to have the same number of channels as the patches
                gf = np.tile(gf, (n_ch, 1, 1))
                # add to filter list
                level_filters.append(template.from_vector(gf.ravel()))
            self.filters.append(level_filters)

    def _extract_patches(self, images, centres):
        patches = []
        for i, c in zip(images, centres):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

    def _compute_filter_responses(self, images, filters):
        # compute responses lists
        images = [self._apply_filters(i, filters) for i in images]
        # flatten out and return list of responses
        return [i for img in images for i in img]

    def _apply_filters(self, image, filters):
        # define extended shape
        ext_h = image.shape[0] + filters[0].shape[0] - 1
        ext_w = image.shape[1] + filters[0].shape[0] - 1
        ext_shape = (filters[0].n_channels, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image)

        # initialize list of responses
        responses = []
        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)
            # compute extended filter fft
            fft_ext_f = fft2(ext_f)

            # compute filter response
            fft_ext_r = fft_ext_f * fft_ext_image
            # compute inverse fft of the filter response
            ext_r = np.real(fftshift(ifft2(fft_ext_r), axes=(-2, -1)))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # transform response to Image
            responses.append(Image(r))

        # return list of responses
        return responses

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = 1
        for level_filters in self.filters[:level]:
            k = 0
            for f in level_filters:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                k += np.real(fft_f.conj() * fft_f)
            kernel *= k
        return Image(fftshift(kernel))

    def compute_network_response(self, image, level=None):
        images = [image]
        for level_filters in self.filters[:level]:
            images = self._compute_filter_responses(images, level_filters)
        return self._list_to_image(images)

    @classmethod
    def _list_to_image(cls, images):
        img = images[0]
        n_ch, h, w = img.pixels.shape
        pixels = np.zeros((len(images) * n_ch, h, w))
        for j, i in enumerate(images):
            ind = j * n_ch
            pixels[ind:ind+n_ch] = i.pixels
        return Image(pixels)

    def compute_kernel_response(self, image, level=None):
        # obtain number of channels
        n_ch, h, w = self.filters[0][0].pixels.shape

        # obtain extended shape
        ext_h = image.shape[0] + h - 1
        ext_w = image.shape[1] + w - 1
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