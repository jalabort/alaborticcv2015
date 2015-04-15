from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from menpo.shape import PointCloud
from menpo.feature import centralize, convert_image_to_dtype
from menpo.visualize import print_dynamic, progress_bar_str


def extract_patches_from_landmarks(image, patch_shape=(7, 7), group=None,
                                   label=None, as_single_array=False,
                                   dtype=np.float32):
    image_cp = image.copy()
    image_cp.pixels_astype(dtype=np.float64)
    patches = image_cp.extract_patches_around_landmarks(
        group=group, label=label, patch_size=patch_shape,
        as_single_array=as_single_array)[:, 0, ...]
    del image_cp
    for j, p in enumerate(patches):
        patches[j] = convert_image_to_dtype(p, dtype=dtype)
    return patches


def extract_patches_from_grid(image, patch_shape=(7, 7), stride=(4, 4),
                              as_single_array=False, dtype=np.float32):
    limit = np.round(np.asarray(patch_shape) / 2)
    image_cp = image.copy()
    image_cp.pixels_astype(dtype=np.float64)
    h, w = image_cp.shape[-2:]
    grid = np.rollaxis(np.mgrid[limit[0]:h-limit[0]:stride[0],
                                limit[1]:w-limit[0]:stride[1]], 0, 3)
    pc = PointCloud(np.require(grid.reshape((-1, 2)), dtype=np.float64))
    patches = image_cp.extract_patches(pc, patch_size=patch_shape,
                                       as_single_array=as_single_array)
    del image_cp
    for j, p in enumerate(patches):
        patches[j] = convert_image_to_dtype(p, dtype=dtype)
    return patches


def extract_patches(images, extract=extract_patches_from_grid,
                    patch_shape=(7, 7), as_single_array=False, flatten=True,
                    dtype=np.float32, verbose=False, string='', **kwargs):
    if verbose:
        string += 'Extracting Patches '
    n_images = len(images)
    if not as_single_array:
        patches = []
        for j, i in enumerate(images):
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(j/n_images, show_bar=True)))

            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            if flatten:
                patches += ps
            else:
                patches.append(ps)
    else:
        ps = extract(images[0], patch_shape=patch_shape, dtype=dtype,
                     as_single_array=as_single_array, **kwargs)
        patches = np.empty((n_images,) + ps.shape)
        patches[0] = ps
        for j, i in enumerate(images[1:]):
            if verbose:
                print_dynamic('{}{}'.format(
                    string, progress_bar_str(j/n_images, show_bar=True)))

            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            patches[j+1] = ps
        if flatten:
            patches = np.reshape(patches, (-1,) + patches.shape[-3:])
    if verbose:
        print_dynamic('{}{}'.format(
            string, progress_bar_str(1, show_bar=True)))
    return patches


def normalize_images(images, norm_func=centralize):
    if norm_func is not None:
        for j, i in enumerate(images):
            images[j] = norm_func(i)
    return images


def normalize_filters(filters, norm_func=centralize):
    if norm_func:
        for j, fs in enumerate(filters):
            filters[j] = normalize_images(fs, norm_func=norm_func)
    return filters


# TODO: should work for both images and ndarrays
def images_to_image(images):
    n_images = len(images)
    n_channels = images[0].n_channels
    pixels = np.empty((n_images * n_channels,) + images[0].shape)
    start = 0
    for i in images:
        finish = start + n_channels
        pixels[start:finish] = i.pixels
        start = finish
    image = images[0].copy()
    image.pixels = pixels
    return image


# TODO: should work for both images and ndarrays
def image_to_images(image):
    n_images, n_channels = image.pixels.shape[:2]
    images = []
    for j in range(n_images):
        i = image.copy()
        i.pixels = image.pixels[j]
        images.append(i)
    return images


def build_grid(shape):
    shape = np.array(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    start = -half_shape
    end = half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)


def generate_gaussian_response(shape, cov):
    mvn = multivariate_normal(mean=np.zeros(2), cov=cov)
    grid = build_grid(shape)
    return mvn.pdf(grid)[None]
