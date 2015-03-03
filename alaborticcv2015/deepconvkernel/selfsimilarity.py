from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from menpo.image import Image

from alaborticcv2015.utils import pad, unpad


class SelfSimDCKArch1():

    def __init__(self, patch_size=(7, 7), mean_centre=True, n_levels=3,
                 padding='constant'):
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.n_levels = n_levels
        self.padding = padding

    def compute_network_response(self, image, level=None, group=None,
                                 label=None):
        if not level:
            level = self.n_levels
        # extract centres
        centres = image.landmarks[group][label]
        level_image = image
        for j in range(level):
            level_filters = self._extract_patches(level_image, centres)
            level_image = self._apply_filters(level_image, level_filters)
        return level_image

    def _extract_patches(self, image, centres):
        patches = image.extract_patches(centres, patch_size=self.patch_size)
        if self.mean_centre:
            # mean centre patches
            for p in patches:
                p.mean_centre_inplace()
        return patches

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
        fft_ext_image = fft2(ext_image, axes=(-2, -1))

        # initialize response
        response = np.zeros(r_ext_shape)
        for j, f in enumerate(filters):

            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)

            # compute extended filter fft
            fft_ext_f = fft2(ext_f, axes=(-2, -1))

            # compute filter response
            fft_ext_r = np.sum(fft_ext_f * fft_ext_image, axis=0)[None]
            # compute inverse fft of the filter response
            ext_r = np.real(fftshift(ifft2(fft_ext_r),
                                            axes=(-2, -1)))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # fill out overall response
            response[j] = r

        # transform response to Image and return it
        return Image(response)

    def compute_kernel_response(self, image, level=None, group=None,
                                label=None):
        if not level:
            level = self.n_levels
        # obtain number of channels
        n_ch, h, w = (image.n_channels,) + self.patch_size

        # obtain extended shape
        ext_h = image.shape[0] + 2 * h
        ext_w = image.shape[1] + 2 * w
        ext_shape = (n_ch, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(image, level=level, group=group,
                                           label=label, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(ifft2(fft_ext_r, axes=(-2, -1)))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)

    def learn_kernel(self, image, level=None, group=None, label=None,
                     ext_shape=None):
        if not level:
            level = self.n_levels
        filters = self.filters(image, level=level, group=group, label=label)
        kernel = self._learn_kernel(filters, ext_shape=ext_shape)
        return Image(fftshift(kernel, axes=(-2, -1)))

    def filters(self, image, level=None, group=None, label=None):
        if not level:
            level = self.n_levels
        # extract centres
        centres = image.landmarks[group][label]

        level_image = image
        level_filters = self._extract_patches(level_image, centres)

        filters = [level_filters]
        for j in range(level-1):
            level_image = self._apply_filters(level_image, level_filters)
            level_filters = self._extract_patches(level_image, centres)
            filters.append(level_filters)

        return filters

    def _learn_kernel(self, filters, ext_shape=None):
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels, axes=(-2, -1))
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels, axes=(-2, -1))
                kernel += np.real(fft_f.conj() * fft_f)

        return kernel


class SelfSimDCKArch2():

    def __init__(self, patch_size=(7, 7), mean_centre=True, n_levels=3,
                 padding='constant'):
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.n_levels = n_levels
        self.padding = padding

    def compute_network_response(self, image, level=None, group=None,
                                 label=None):
        if not level:
            level = self.n_levels
        # extract centres
        centres = image.landmarks[group][label]
        level_images = [image]
        for j in range(level):
            level_filters = self._extract_patches(level_images, centres)
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
        return self._list_to_image(level_images)

    def _extract_patches(self, images, centres):
        patches = []
        for i in images:
            i_patches = i.extract_patches(centres, patch_size=self.patch_size)
            if self.mean_centre:
                # mean centre patches
                for p in i_patches:
                    p.mean_centre_inplace()
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

    def _compute_filter_responses(self, images, filters):
        # compute responses lists
        images = [self._apply_filters(i, filters) for i in images]
        # flatten out and return list of responses
        return [i for img in images for i in img]

    def _apply_filters(self, image, filters):
        # define extended shape
        ext_h = image.shape[0] + 2 * filters[0].shape[0]
        ext_w = image.shape[1] + 2 * filters[0].shape[1]
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
            fft_ext_f = fft2(ext_f, axes=(-2, -1))

            # compute filter response
            fft_ext_r = fft_ext_f * fft_ext_image
            # compute inverse fft of the filter response
            ext_r = np.real(fftshift(ifft2(
                fft_ext_r, axes=(-2, -1)), axes=(-2, -1)))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # transform response to Image
            responses.append(Image(r))

        # return list of responses
        return responses

    @classmethod
    def _list_to_image(cls, images):
        img = images[0]
        n_ch, h, w = img.pixels.shape
        pixels = np.zeros((len(images) * n_ch, h, w))
        for j, i in enumerate(images):
            ind = j * n_ch
            pixels[ind:ind+n_ch] = i.pixels
        return Image(pixels)

    def compute_kernel_response(self, image, level=None, group=None,
                                label=None):
        if not level:
            level = self.n_levels
        # obtain number of channels
        n_ch, h, w = (image.n_channels,) + self.patch_size

        # obtain extended shape
        ext_h = image.shape[0] + 2 * h
        ext_w = image.shape[1] + 2 * w
        ext_shape = (n_ch, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(image, level=level, group=group,
                                           label=label, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(ifft2(fft_ext_r, axes=(-2, -1)))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)

    def learn_kernel(self, image, level=None, group=None, label=None,
                     ext_shape=None):
        if not level:
            level = self.n_levels
        filters = self.filters(image, level=level, group=group, label=label)

        kernel = 1
        for level_filters in filters:
            k = 0
            for f in level_filters:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels, axes=(-2, -1))
                k += np.real(fft_f.conj() * fft_f)
            kernel *= k
        return Image(fftshift(kernel, axes=(-2, -1)))

    def filters(self, image, level=None, group=None, label=None):
        if not level:
            level = self.n_levels
        # extract centres
        centres = image.landmarks[group][label]

        level_images = [image]
        level_filters = self._extract_patches(level_images, centres)

        filters = [level_filters]
        for j in range(level-1):
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            level_filters = self._extract_patches(level_images, centres)
            filters.append(level_filters)

        return filters