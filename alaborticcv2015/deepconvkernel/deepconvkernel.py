from __future__ import division
import numpy as np

from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math import log_gabor
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


def learn_gabor_filters(patches, num_scales=2, num_orientations=4,
                        min_wavelength=2, scaling_constant=2,
                        center_sigma=0.65, d_phi_sigma=1.3):
    # build log gabor
    _, _, _, gabor_filters = log_gabor(patches[0].pixels[0],
                                       num_scales=num_scales,
                                       num_orientations=num_orientations,
                                       min_wavelength=min_wavelength,
                                       scaling_constant=scaling_constant,
                                       center_sigma=center_sigma,
                                       d_phi_sigma=d_phi_sigma)
    n_ch = patches[0].n_channels
    filters = []
    for gf in gabor_filters:
        # convert to spatial domain
        gf = np.real(np.fft.ifft2(gf[None], axes=(-2, -1)))
        # tile to have the same number of channels as the patches
        gf = np.tile(gf, (n_ch, 1, 1))
        # add to filter list
        filters.append(patches[0].from_vector(gf.ravel()))

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


# Deep learnable convolutional kernels ----------------------------------------

class DeepLearnConvKernelArch1():

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
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
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


class DeepLearnConvKernelArch2():

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
                k += np.real(fft_f.conj() * fft_f)
            kernel *= k
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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


# Learn discriminative correlation filters ------------------------------------

def learn_dc_filters(patches, mean_centre=True):
    if mean_centre:
        # mean centre patches
        for p in patches:
            p.mean_centre_inplace()
    # learn pca model
    cf = CFModel(patches)
    # grab filters
    filters = [cf.template_instance.from_vector(f) for f in cf.filters]
    return filters


# Deep discriminative convolutional kernels -----------------------------------

class DeepDiscConvKernelArch1():

    def __init__(self, learn_filters=learn_dc_filters, n_levels=3,
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
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
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


# Deep self-similarity convolutional kernels ----------------------------------

class DeepSelfSimConvKernelArch1():

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
            ext_r = np.real(np.fft.fftshift(np.fft.ifft2(fft_ext_r),
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
        fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(image, level=level, group=group,
                                           label=label, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = np.fft.fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(np.fft.ifft2(fft_ext_r, axes=(-2, -1)))

        # crop deep response to original image size and return it
        r = unpad(ext_r, image.shape)
        return Image(r)

    def learn_kernel(self, image, level=None, group=None, label=None,
                     ext_shape=None):
        if not level:
            level = self.n_levels
        filters = self.filters(image, level=level, group=group, label=label)
        kernel = self._learn_kernel(filters, ext_shape=ext_shape)
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
                kernel += np.real(fft_f.conj() * fft_f)

        return kernel


class DeepSelfSimConvKernelArch2():

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
        fft_ext_image = np.fft.fft2(ext_image)

        # initialize list of responses
        responses = []
        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=self.padding)
            # compute extended filter fft
            fft_ext_f = np.fft.fft2(ext_f, axes=(-2, -1))

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
        fft_ext_image = np.fft.fft2(ext_image, axes=(-2, -1))

        # compute deep convolutional kernel
        fft_ext_kernel = self.learn_kernel(image, level=level, group=group,
                                           label=label, ext_shape=ext_shape)

        # compute deep response by convolving image with sqrt of deep kernel
        fft_ext_r = np.fft.fftshift(
            fft_ext_kernel.pixels**0.5, axes=(-2, -1)) * fft_ext_image

        # compute inverse fft of deep response
        ext_r = np.real(np.fft.ifft2(fft_ext_r, axes=(-2, -1)))

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
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
                k += np.real(fft_f.conj() * fft_f)
            kernel *= k
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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


# Deep log-gabor convolutional kernels ----------------------------------------

class DeepLogGaborConvKernelArch1():

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
            gf = np.real(np.fft.fftshift(np.fft.ifft2(gf[None],
                                                      axes=(-2, -1))))
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
                gf = np.real(np.fft.fftshift(np.fft.ifft2(gf[None],
                                                          axes=(-2, -1))))
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
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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
                kernel += np.real(fft_f.conj() * prev_kernel[j][None] * fft_f)
        else:
            kernel = 0
            for f in filters[0]:
                if ext_shape:
                    f_pixels = pad(f.pixels, ext_shape, mode=self.padding)
                else:
                    f_pixels = f.pixels
                fft_f = np.fft.fft2(f_pixels, axes=(-2, -1))
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


class DeepLogGaborConvKernelArch2():

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
                gf = np.real(np.fft.fftshift(np.fft.ifft2(gf[None],
                                                          axes=(-2, -1))))
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
                k += np.real(fft_f.conj() * fft_f)
            kernel *= k
        return Image(np.fft.fftshift(kernel, axes=(-2, -1)))

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
