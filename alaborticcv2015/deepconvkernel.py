from __future__ import division
import numpy as np

from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math.decomposition import _batch_ica, negentropy_logcosh
from menpo.image import Image

from menpo.visualize import print_dynamic, progress_bar_str


# Learn component analysis filters --------------------------------------------

def learn_pca_filters(patches, n_filters=8, mean_centre=True):
    if mean_centre:
        # mean centre patches
        for p in patches:
            p.mean_centre_inplace()
    # learn pca model
    pca = PCAModel(patches)
    # set active number of components
    pca.n_active_components = n_filters
    # grab filters
    filters = [pca.template_instance.from_vector(pc) for pc in pca.components]
    return filters


def learn_ica_filters(patches, n_filters=8, mean_centre=True,
                      algorithm=_batch_ica, negentropy=negentropy_logcosh,
                      max_iters=500):
    if mean_centre:
        # mean centre patches
        for p in patches:
            p.mean_centre_inplace()
    # learn ica model
    ica = ICAModel(patches, algorithm=algorithm, negentropy_approx=negentropy,
                   n_components=n_filters, max_iters=max_iters)
    # grab filters
    filters = [ica.template_instance.from_vector(ic) for ic in ica.components]
    return filters


def learn_nmf_filters(patches, n_filters=8, max_iters=500, verbose=False,
                      **kwargs):
    # learn nmf model
    nmf = NMFModel(patches, n_components=n_filters, max_iters=max_iters,
                   verbose=verbose, **kwargs)
    # grab filters
    filters = [nmf.template_instance.from_vector(nf) for nf in nmf.components]
    return filters


# Image padding ---------------------------------------------------------------

def pad(pixels, ext_shape, mode='constant'):
    _, h, w = pixels.shape
    h_margin = np.floor((ext_shape[1] - h) / 2)
    w_margin = np.floor((ext_shape[2] - w) / 2)

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[1]:
        h_margin2 = h_margin + 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[2]:
        w_margin2 = w_margin + 1

    pad_width = ((0, 0), (h_margin, h_margin2), (w_margin, w_margin2))
    return np.lib.pad(pixels, pad_width, mode=mode)


def unpad(pixels, red_shape):
    _, h, w = pixels.shape
    h_margin = (h - red_shape[0]) / 2
    w_margin = (w - red_shape[1]) / 2
    return pixels[:, h_margin-1:-h_margin-1, w_margin-1:-w_margin-1]


# Deep convolutional kernel architectures -------------------------------------

class DeepConvKernelArch1():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
                 n_filters=8, patch_size=(7, 7), mean_centre=True,
                 padding='constant'):
        self._learn_filters = learn_filters
        self.n_levels = n_levels
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.padding = padding

    def learn_network(self, training_images, group=None, label=None,
                      verbose=False, **kwargs):
        # extract centers
        centers = [i.landmarks[group][label] for i in training_images]
        # initialize level_image and list of filters
        level_images = training_images
        self.filters = []
        for j in range(self.n_levels):
            if verbose:
                print_dynamic(
                    '- Computing network: {}'.format(
                        progress_bar_str((j + 1.) / self.n_levels,
                                         show_bar=True)))
            # extract level patches
            level_patches = self._extract_patches(level_images, centers)
            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            # save level filters
            self.filters.append(level_filters)

    def _extract_patches(self, images, centers):
        patches = []
        for i, c in zip(images, centers):
            i_patches = i.extract_patches(c, patch_size=self.patch_size)
            patches.append(i_patches)
        return [i for i_patches in patches for i in i_patches]

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
        fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))

        # initialize response
        response = np.zeros(r_ext_shape)
        for j, f in enumerate(filters):

            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)

            # compute extended filter fft
            fft_ext_f = np.fft.fft2(ext_f, axes=(-2, -1))

            # compute filter response
            fft_ext_r = np.sum(fft_ext_f * fft_ext_image, axis=0)[None]
            # compute inverse fft of the filter response
            ext_r = np.fft.fftshift(np.real(np.fft.ifft2(fft_ext_r)),
                                    axes=(-2, -1))

            # crop response to original image size
            r = unpad(ext_r, image.shape)
            # fill out overall response
            response[j] = r

        # transform response to Image and return it
        return Image(response)

    def learn_kernel(self, level=None, ext_shape=None):
        kernel = self._learn_kernel(self.filters[:level], ext_shape=ext_shape)
        return Image(np.real(np.fft.fftshift(kernel, axes=(-2, -1))))

    def _learn_kernel(self, filters, ext_shape=None):
        if len(filters) > 1:
            prev_kernel = self._learn_kernel(filters[1:], ext_shape=ext_shape)
            kernel = 0
            for j, f in enumerate(filters[0]):
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
                kernel += fft_f.conj() * prev_kernel[j][None] * fft_f
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
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
        ext_h = image.shape[0] + 2 * h
        ext_w = image.shape[1] + 2 * w
        ext_shape = (n_ch, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = np.fft.fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(np.fft.ifft2(fft_ext_r, axes=(-2, -1)))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)

    # def _compute_network_response(self, image, filters):
    #     for
    #     if len(filters) > 1:
    #         response = self._compute_network_response(image, filters[:-2])
    #         return
    #     else:
    #         response = self._apply_filters(image, filters[0])
    #
    #     return response

    # def compute_kernel_response(self, image):
    #     # obtain number of channels
    #     n_ch = self.kernel.n_channels
    #
    #     # obtain extended shape
    #     ext_h = image.shape[0] + 2 * self.kernel.shape[0]
    #     ext_w = image.shape[1] + 2 * self.kernel.shape[1]
    #     ext_shape = (n_ch, ext_h, ext_w)
    #
    #     # extend image
    #     ext_image = pad(image.pixels, ext_shape, mode=self.padding)
    #     # compute extended image fft
    #     fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))
    #
    #     # extend deep convolutional kernel
    #     ext_kernel = pad(self.kernel.pixels, ext_shape)
    #     # compute extended kernel fft
    #     fft_ext_kernel = np.fft.fft2(ext_kernel, axes=(-2, -1))
    #
    #     # compute deep response by convolving image with sqrt of deep kernel
    #     fft_ext_r = np.abs(fft_ext_kernel)**0.5 * fft_ext_image
    #     # compute inverse fft of deep response
    #     ext_r = np.real(np.fft.ifft2(fft_ext_r))
    #
    #     # crop deep response to original image size and return it
    #     r = unpad(ext_r, image.shape)
    #     return Image(r)


class DeepConvKernelArch2():

    def __init__(self, learn_filters=learn_pca_filters, n_levels=3,
                 n_filters=8, patch_size=(7, 7), mean_centre=True,
                 padding='constant'):
        self._learn_filters = learn_filters
        self.n_levels = n_levels
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.padding = padding

    def learn_network(self, training_images, group=None, label=None,
                      verbose=False, **kwargs):
        # extract centers
        centers = [i.landmarks[group][label] for i in training_images]
        # initialize level_image and list of filters
        level_images = training_images
        self.filters = []
        for j in xrange(self.n_levels):
            if verbose:
                print_dynamic(
                    '- Computing network: {}'.format(
                        progress_bar_str((j + 1.) / self.n_levels,
                                         show_bar=True)))
            # extract level patches
            level_patches = self._extract_patches(level_images, centers)
            # learn level filters
            level_filters = self._learn_filters(level_patches, self.n_filters,
                                                **kwargs)
            # compute level responses lists
            level_images = self._compute_filter_responses(level_images,
                                                          level_filters)
            # save level filters
            self.filters.append(level_filters)

    def _extract_patches(self, images, centers):
        patches = []
        for i, c in zip(images, centers):
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
        ext_h = image.shape[0] + 2 * filters[0].shape[0]
        ext_w = image.shape[1] + 2 * filters[0].shape[1]
        ext_shape = (filters[0].n_channels, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = np.fft.fft2(ext_image)

        # initialize list of responses
        responses = []
        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)
            # compute extended filter fft
            fft_ext_f = np.fft.fft2(ext_f)

            # compute filter response
            fft_ext_r = fft_ext_f * fft_ext_image
            # compute inverse fft of the filter response
            ext_r = np.real(np.fft.fftshift(np.fft.ifft2(
                fft_ext_r, axes=(-2, -1)), axes=(-2, -1)))

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
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
                k += fft_f.conj() * fft_f
            kernel *= k
        return Image(np.real(np.fft.fftshift(kernel, axes=(-2, -1))))

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
        ext_h = image.shape[0] + 2 * h
        ext_w = image.shape[1] + 2 * w
        ext_shape = (n_ch, ext_h, ext_w)

        # extend image
        ext_image = pad(image.pixels, ext_shape, mode=self.padding)
        # compute extended image fft
        fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(level=level, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = np.fft.fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(np.fft.ifft2(fft_ext_r, axes=(-2, -1)))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)

