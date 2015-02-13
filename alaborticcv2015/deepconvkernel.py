from __future__ import division
import numpy as np

from menpo.model import PCAModel, ICAModel, NMFModel
from menpo.math.decomposition import _batch_ica, negentropy_logcosh
from menpo.image import Image


class DeepConvKernel():

    def __init__(self, filter_cls=PCAModel, n_levels=3, n_filters=8,
                 patch_size=(17, 17), mean_centre=True, mode='constant'):
        self.filter_cls = filter_cls
        self.n_levels = n_levels
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.mean_centre = mean_centre
        self.mode = mode

    def apply_network(self, images):
        for i in images:
            for filters in self.net_filters:
                for f in filters:
                    fmap = convolve(i, f)

def convolve(image, filter):



def learn_network(images, centres, filter_cls=PCAModel, n_levels=3,
                  n_filters=8, patch_size=(17, 17), mean_centre=True,
                  save_responses=False, mode='constant'):

    # initialize list of filters
    filters_list =[]
    # initialize list of responses
    responses_list = []

    for j in xrange(n_levels):
        # extract jth level image patches
        patches = extract_patches(images, centres, patch_size, mean_centre)
        # learn jth level filters from patches
        filters = learn_filters(patches, filter_cls, n_filters)
        # compute jth level responses by applying filters to images
        images = apply_filters(images, filters, mode=mode)

        # save jth level filters
        filters_list.append(filters)
        if save_responses:
            # save jth level responses
            responses_list.append(images)

    return filters_list, responses_list




# def learn_filters(patches, filter_cls=PCAModel, n_filters=8):
#     pca = PCAModel(patches)
#     pca.n_active_components = n_filters
#     filters = [pca.template_instance.from_vector(pc) for pc in pca.components]
#     return filters

# def learn_filters(patches, filter_cls=ICAModel, n_filters=8):
#     ica = ICAModel(patches, algorithm=_batch_ica, negentropy_approx=negentropy_logcosh,
#                    n_components=n_filters, max_iters=500)
#     filters = [ica.template_instance.from_vector(ic) for ic in ica.components]
#     return filters

def learn_filters(patches, filter_cls=NMFModel, n_filters=8):
    nmf = NMFModel(patches, n_components=n_filters, max_iters=500, verbose=True,
                   alpha=2)
    filters = [nmf.template_instance.from_vector(nf) for nf in nmf.components]
    return filters

# from sklearn.decomposition import NMF

# def learn_filters(patches, filter_cls=NMFModel, n_filters=8):
#     nmf = NMF(n_components=n_filters)
#     patches_pixels = [i.as_vector() for i in patches]
#     X = np.asarray(patches_pixels).copy()
#     nmf.fit(X)
#     filters = [nmf.template_instance.from_vector(nf) for nf in nmf.components_]
#     return filters


def apply_filters(images, filters, mode='constant'):
    # obtain number of filters, number of channels, shape and x and y limits
    n_filters = len(filters)
    n_ch = filters[0].n_channels
    f_shape = filters[0].shape
    f_limy, f_limx = np.asarray(f_shape)
    n_elements = f_limy * f_limx

    # initialize list of responses
    responses = []
    for i in images:
        # obtain filters n_channels, shape and x and y limits
        i_shape = i.shape

        # obtain extended shape
        ext_h = i.shape[0] + 2 * f_limy
        #ext_h = ext_h if ext_h % 2 == 0 else ext_h + 1
        ext_w = i.shape[1] + 2 * f_limx
        #ext_w = ext_w if ext_w % 2 == 0 else ext_w + 1
        ext_shape = (n_ch, ext_h, ext_w)

        # obtain response extended shape
        r_ext_shape = (n_filters, i.shape[0], i.shape[1])

        # extend image
        ext_i = pad(i.pixels, ext_shape, mode=mode)

        # compute extended image fft
        fft_ext_i = np.fft.fft2(ext_i)

        # initialize response
        response = np.zeros(r_ext_shape)

        for j, f in enumerate(filters):
            # extend filter
            ext_f = pad(f.pixels, ext_shape, mode=mode)

            # compute extended filter fft
            fft_ext_f = np.fft.fft2(ext_f)

            # compute filter response by convolving image and filter
            fft_ext_r = np.sum(fft_ext_f * fft_ext_i, axis=0)

            # compute inverse fft of the previous response
            ext_r = np.fft.fftshift(np.real(np.fft.ifft2(fft_ext_r)), axes=(-2, -1))

            # crop response to original image size
            r = unpad(ext_r, i_shape)

            # fill out overall response
            response[j] = r

        # tranform response to Image and append it to the list
        response = Image(response.reshape((-1,) + i_shape))
        responses.append(response)

    return responses


def extract_patches(images, centres, patch_size, mean_centre):
    image_patches = []
    for (i, c) in zip(images, centres):
        patches = i.extract_patches(c, patch_size=patch_size)
        for p in patches:
            #if mean_centre:
            #    p.mean_centre_inplace()
            image_patches.append(p)
    return image_patches


def pad(pixels, ext_shape, mode='constant'):
    _, h, w = pixels.shape
    h_margin = (ext_shape[1] - h) / 2
    w_margin = (ext_shape[2] - w) / 2

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[1]:
        h_margin2 = h_margin + 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[2]:
        w_margin2 = w_margin + 1

    pad_width = ((0, 0), (h_margin, h_margin2), (w_margin, w_margin2))
    return np.lib.pad(pixels, pad_width, mode=mode)


def unpad(pixels, red_shape, mode='constant'):
    h, w = pixels.shape
    h_margin = (h - red_shape[0]) / 2
    w_margin = (w - red_shape[1]) / 2
    return pixels[h_margin-1:-h_margin-1, w_margin-1:-w_margin-1]


# def augment_kernel_shape(k, shape, mode='constant'):
#     n_ch = k.n_channels
#     k_shape = k.shape
#     k_limy, k_limx = np.asarray(k_shape)

#     # obtain extended shape
#     ext_h = shape[0] + 2 * k_limy
#     ext_w = shape[1] + 2 * k_limx
#     ext_shape = (n_ch, ext_h, ext_w)

#     spatial_k = np.real(np.fft.ifft2(k.pixels))
#     ext_k = pad(spatial_k, ext_shape, mode=mode)
#     # compute extended filter fft
#     fft_ext_k = np.abs(np.fft.fftshift(np.fft.fft2(ext_k)))

#     return Image(fft_ext_k)


def compute_kernels(filters, ext_shape=None, mode='constant'):
    kernels = []
    deep_kernels = []
    dK = 1
    for fs in filters:
        K = 0
        for f in fs:
            if ext_shape:
                f_pixels = pad(f.pixels, ext_shape, mode=mode)
            fft_f = np.fft.fft2(f_pixels)
            K += np.abs(fft_f)
        dK *= np.sum(K, axis=0)
        kernels.append(Image(np.fft.fftshift(K, axes=(-2, -1))))
        deep_kernels.append(Image(np.fft.fftshift(dK[None, ...], axes=(-2, -1))))

    return deep_kernels, kernels


def compute_deep_kernel(filters, ext_shape=None, mode='constant'):
    dK = 1
    for fs in filters:
        K = 0
        for f in fs:
            if ext_shape:
                f_pixels = pad(f.pixels, ext_shape, mode=mode)
            fft_f = np.fft.fft2(f_pixels)
            K += np.abs(fft_f)
        dK *= np.sum(K, axis=0)
    return dK[None, ...]


def deep_convolution(i, filters, mode='constant'):
    n_ch = 0
    f_shape = filters[0][0].shape
    f_limy, f_limx = np.asarray(f_shape)

    # obtain extended shape
    ext_h = i.shape[0] + 2 * f_limy
    ext_w = i.shape[1] + 2 * f_limx
    ext_shape = (n_ch, ext_h, ext_w)

    # extend image
    ext_i = pad(i.pixels, ext_shape, mode=mode)
    # compute extended image fft
    fft_ext_i = np.fft.fft2(ext_i)

    # compute deep kernel
    dK = compute_deep_kernel(filters, ext_shape)

    # compute deep response by convolving image and deep kernel
    fft_ext_r = np.sum(fft_ext_i * dK**0.5, axis=0)

    # compute inverse fft of the previous response
    ext_r = np.real(np.fft.ifft2(fft_ext_r))

    # crop response to original image size
    r = unpad(ext_r, i.shape)

    return Image(r)