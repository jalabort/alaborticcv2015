from __future__ import division
import numpy as np
from menpo.feature import centralize, convert_image_to_dtype
from menpo.shape import PointCloud


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
                    patch_shape=(7, 7), as_single_array=False,
                    dtype=np.float32, **kwargs):
    if not as_single_array:
        patches = []
        for i in images:
            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            patches.append(ps.reshape((-1,) + ps.shape[-3:]))
    else:
        ps = extract(images[0], patch_shape=patch_shape, dtype=dtype,
                     as_single_array=as_single_array, **kwargs)
        patches = ps.reshape((-1,) + ps.shape[-3:])
        for i in images:
            ps = extract(i, patch_shape=patch_shape, dtype=dtype,
                         as_single_array=as_single_array, **kwargs)
            ps = ps.reshape((-1,) + ps.shape[-3:])
            patches = np.concatenate((patches, ps), axis=0)
    return patches


def normalize_patches(patches, norm_func=centralize):
    if norm_func is not None:
        for j, p in enumerate(patches):
            patches[j] = norm_func(p)
    return patches
