from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal.signaltools import _centered, _next_regular


from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math.decomposition.ica import _batch_ica, negentropy_logcosh
from menpo.image import Image
from menpo.visualize import print_dynamic, progress_bar_str

from alaborticcv2015.utils import pad, unpad


def learn_pca_filters(patches, n_filters=8, mean_centre=True):
    if mean_centre:
        new_patches = []
        # mean centre patches
        for p in patches:
            pp = p.copy()
            pp.mean_centre_inplace()
            new_patches.append(pp)
        patches = new_patches
    # learn pca model
    pca = PCAModel(patches)
    # set active number of components
    pca.n_active_components = n_filters
    # grab filters
    filters = [pca.template_instance.from_vector(pc)
               for pc in pca.components]
    return filters


def learn_ica_filters(patches, n_filters=8, mean_centre=True,
                      algorithm=_batch_ica, negentropy=negentropy_logcosh,
                      max_iters=500):
    if mean_centre:
        new_patches = []
        # mean centre patches
        for p in patches:
            pp = p.copy()
            pp.mean_centre_inplace()
            new_patches.append(pp)
        patches = new_patches

    # learn ica model
    ica = ICAModel(patches, algorithm=algorithm, negentropy_approx=negentropy,
                   n_components=n_filters, max_iters=max_iters)
    # grab filters
    filters = [ica.template_instance.from_vector(ic)
               for ic in ica.components]
    return filters


def learn_nmf_filters(patches, n_filters=8, max_iters=500, verbose=False,
                      **kwargs):
    # learn nmf model
    nmf = NMFModel(patches, n_components=n_filters, max_iters=max_iters,
                   verbose=verbose, **kwargs)
    # grab filters
    filters = [nmf.template_instance.from_vector(nf)
               for nf in nmf.components]
    return filters


class GenerativeDCKArch1():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
                 n_filters=8, patch_size=(7, 7), mean_centre=True,
                 crop=False, padding='constant'):
        self._learn_filters = learn_filters
        self.n_levels = n_levels
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.crop = crop
        self.padding = padding

    def learn_network(self, images, group=None, label=None,
                      verbose=False, **kwargs):
        if verbose:
            string = '- Learning network'
        # extract centres
        centres = [i.landmarks[group][label] for i in images]
        # initialize level_image and list of filters
        level_images = images
        self.filters = []
        for j in range(self.n_levels):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_levels, show_bar=True)))
            # extract level patches
            level_patches = self._extract_patches(level_images, centres)
            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(
                level_images, level_filters, centres)
            # save level filters
            self.filters.append(level_filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

    def _extract_patches(self, images, centres):
        patches = []
        for i, c in zip(images, centres):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

    def _compute_filter_responses(self, images, filters, centres):
        # compute  and return list of responses
        return [self._apply_filters(i, filters, c)
                for (i, c) in zip(images, centres)]

    def _apply_filters(self, image, filters, centre=False):
        # define extended shape
        ext_h = image.shape[0] + filters[0].shape[0] - 1
        ext_w = image.shape[1] + filters[0].shape[1] - 1
        ext_shape = (ext_h, ext_w)
        # define response extended shape
        if self.crop:
            r_shape = (len(filters), image.shape[0], image.shape[1])
        else:
            r_shape = (len(filters),) + ext_shape

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))

        # initialize response
        response = np.zeros(r_shape)
        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)
            # compute extended filter fft
            fft_ext_f = fftshift(fft2(ext_f), axes=(-2, -1))
            # compute filter response
            fft_ext_r = np.sum(fft_ext_f * fft_ext_image, axis=0)
            # compute inverse fft of the filter response
            r = np.real(ifftshift(ifft2(ifftshift(fft_ext_r,
                                                 axes=(-2, -1))),
                                 axes=(-2, 1)))[None]
            if self.crop:
                r = unpad(r, image.shape)
            else:
                if centre:
                    centre.points += np.asarray((filters[0].shape[0] - 1,
                                                 filters[0].shape[1] - 1))
            # fill out overall response
            response[j] = r

        # transform response to Image and return it
        return Image(response)

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = self._learn_kernel(self.filters[:level], ext_shape=ext_shape)
        return Image(np.real(kernel))

    def _learn_kernel(self, filters, ext_shape=None):
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fftshift(fft2(f_pixels), axes=(-2, -1))
                kernel += fft_f.conj() * prev_kernel[j] * fft_f
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fftshift(fft2(f_pixels), axes=(-2, -1))
                kernel += fft_f.conj() * fft_f

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
        ext_shape = (ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fft_ext_kernel.pixels**0.5 * fft_ext_image

        # compute inverse fft of deep response
        r = np.real(ifft2(ifftshift(fft_ext_r, axes=(-2, -1))))
        r = unpad(r, image.shape)

        return Image(r)


class GenerativeDCKArch2():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
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
        # initialize level_image and list of filters
        level_images = images
        self.filters = []
        for j in xrange(self.n_levels):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_levels, show_bar=True)))
            # extract level patches
            level_patches = self._extract_patches(level_images, centres)
            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            # save level filters
            self.filters.append(level_filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

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
        ext_w = image.shape[1] + filters[0].shape[1] - 1
        ext_shape = (filters[0].n_channels, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))

        # initialize list of responses
        responses = []
        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)
            # compute extended filter fft
            fft_ext_f = fftshift(fft2(ext_f), axes=(-2, -1))

            # compute filter response
            fft_ext_r = fft_ext_f * fft_ext_image
            # compute inverse fft of the filter response
            ext_r = fftshift(np.real(ifft2(ifftshift(fft_ext_r,
                                                     axes=(-2, -1)))),
                             axes=(-2, -1))

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
                fft_f = fftshift(fft2(f_pixels), axes=(-2, -1))
                k += fft_f.conj() * fft_f
            kernel *= k
        return Image(kernel)

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
        fft_ext_image = fftshift(fft2(ext_image), axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = fft_ext_kernel.pixels**0.5 * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(ifft2(ifftshift(fft_ext_r, axes=(-2, -1))))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)


class GenerativeDCKArch3():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
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
        # initialize level_image and list of filters
        level_images = images
        self.filters = []
        for j in range(self.n_levels):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_levels, show_bar=True)))
            # extract level patches
            level_patches = self._extract_patches(level_images, centres)
            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            # save level filters
            self.filters.append(level_filters)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

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
        N = np.sqrt(ext_h * ext_w)
        ext_shape = (filters[0].n_channels, ext_h, ext_w)
        # define response extended shape
        r_ext_shape = (image.n_channels, image.shape[0], image.shape[1])

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = fft2(ext_image)

        # initialize response
        response = 0
        for f in filters:
            n_ch = f.shape[0]

            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)

            # compute extended filter fft
            fft_ext_f = fft2(ext_f)
            # compute filter response
            fft_ext_r = fft_ext_f * fft_ext_image
            # compute inverse fft of the filter response
            ext_r = fftshift(np.real(ifft2(fft_ext_r)), axes=(-2, -1))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # fill out overall response
            response += r

        # transform response to Image and return it
        return Image(response)

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = self._learn_kernel(self.filters[:level], ext_shape=ext_shape)
        return Image(fftshift(kernel, axes=(-2, -1)))

    def _learn_kernel(self, filters, ext_shape=None):
        _, ext_h, ext_w = ext_shape
        N = ext_h * ext_w
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                n_ch = f.shape[0]
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = fft2(f_pixels)
                kernel += fft_f * prev_kernel[j] * fft_f
        else:
            kernel = 0
            for f in filters[0]:
                n_ch = f.shape[0]
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
        ext_h = image.shape[0] + h - 1
        ext_w = image.shape[1] + w - 1
        N = np.sqrt(ext_h * ext_w)
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


class GenerativeDCKArch4():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
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
        # initialize level_image and list of filters
        level_images = images
        self.filters = []
        for j in xrange(self.n_levels):
            if verbose:
                print_dynamic('{}: {}'.format(
                    string, progress_bar_str(j/self.n_levels, show_bar=True)))

            # extract level patches
            level_patches = self._extract_patches(level_images, centres)

            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # save level filters
            self.filters.append(level_filters)

            # compute level responses lists
            level_images = self._compute_kernel_responses(images)

        if verbose:
            print_dynamic('{}: Done!\n'.format(string))

    def _extract_patches(self, images, centres):
        patches = []
        for i, c in zip(images, centres):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

    def _compute_kernel_responses(self, images):
        # compute and return kernel responses
        return [self.compute_kernel_response(i) for i in images]

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
        r = unpad(ext_r, image.pixels.shape)
        return Image(r)

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
        return Image(fftshift(kernel, axes=(-2, -1)))

    def compute_network_response(self, image, level=None):
        images = [image]
        for level_filters in self.filters[:level]:
            images = self._compute_filter_responses(images, level_filters)
        return self._list_to_image(images)

    def _compute_filter_responses(self, images, filters):
        # compute responses lists
        images = [self._apply_filters(i, filters) for i in images]
        # flatten out and return list of responses
        return [i for img in images for i in img]

    def _apply_filters(self, image, filters):
        # define extended shape
        ext_h = image.shape[0] + filters[0].shape[0] - 1
        ext_w = image.shape[1] + filters[0].shape[1] - 1
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

    @classmethod
    def _list_to_image(cls, images):
        img = images[0]
        n_ch, h, w = img.pixels.shape
        pixels = np.zeros((len(images) * n_ch, h, w))
        for j, i in enumerate(images):
            ind = j * n_ch
            pixels[ind:ind+n_ch] = i.pixels
        return Image(pixels)