# -*- coding: utf-8 -*-
"""Utils for visualizations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def patch_to_img(stack, patches):
    """Draws the patches multiplied by the factor of the stack to a black
    image.

    This function draws a set of `patches` to an initially black image whereas
    each patch is scaled by the respective factor of the `stack`. Drawing
    means addition in this case. Hence, this method is used to realize the
    reconstruction visualizations.

    The `stack` is assumed as (rows, cols, number_of_patches) where each
    element resembles the scaling factor for the patch at the respective
    position. The `patches` have a shape of
    (rows_patch, cols_patch, number_of_patches). The function accepts
    2-dimensional stacks and patches if only one patch is given.

    # Input:
        stack: Numpy array of floats, specifying the scaling value
            (probability) for each patch at each position. The shape
            is (rows, cols, number_of_patches) or (rows, cols) if only one
            patch is given.
        patches: Numpy array of patches, specifying the patch images. The shape
            is (rows_patch, cols_patch, number_of_patches) or
            (rows_patch, cols_patch) if only one patch is given.

    # Output:
        Numpy array of floats of shape
        (rows + rows_patch - 1, cols + cols_patch - 1)
    """
    # initialize black image
    img = np.zeros((stack.shape[0] + patches.shape[0] - 1,
                    stack.shape[1] + patches.shape[1] - 1))

    # expand shape if input dimension is 2
    if patches.ndim == 2:
        patches = np.expand_dims(patches, -1)

    if stack.ndim == 2:
        stack = np.expand_dims(stack, -1)

    # add the scaled patches
    for m in range(stack.shape[0]):
        for n in range(stack.shape[1]):
            img[m:m + patches.shape[0], n:n + patches.shape[1]] += \
                np.sum(stack[m:m + 1, n:n + 1] * patches, -1)

    return img


def pixel_counts(spatial_size, receptive_field):
    """Counts how often a receptive field covers a pixel position in an
    image with the respective spatial size.

    Assume a 2-dimensional convolution with kernel size equal to
    `receptive_field` and an input image of `spatial_size`. Moreover,
    we apply no padding, no dilation and a stride of 1x1. The function
    computes how often the kernel covers a pixel in the image and, hence,
    how often was a pixel in the input image used for a computation. This
    function is used to determine the scaling matrix for the reconstructions.

    # Inputs:
        spatial_size: 2-dimensional list/tuple of integers, specifying the
            spatial size of the image.
        receptive_field: 2-dimensional list/tuple of integers, specifying the
            spatial size of the kernel.

    # Output:
        2-dimensional numpy array of floats with shape `spatial_size` where
        each element is the respective count at a certain position.
    """
    patch = np.ones((receptive_field[0], receptive_field[1]))
    stack = np.ones((spatial_size[0], spatial_size[1]))

    return patch_to_img(stack, patch)


def make_uint8_img(x):
    """Converts an image into an uint8 3-dimensional image.

    The image is assumed as a numpy array. The function accepts a 2 or
    3-dimensional image of float or uint8 values. If the values are float,
    then the method assumes that the values are in the range [0,1] and
    converts them to uint8 by multiplying them with 255 first. Moreover,
    if the input dimension is 2, then the method automatically expands the
    dimension and repeats the image to three channels.

    # Inputs:
        x: Input image as numpy array of floats in the range [0,1] or unit8
            values. The shape could be 2 or 3-dimensional.

    # Output:
        Input image as numpy array of uint8 values with shape (rows, cols, 3).
    """
    if x.ndim == 2:
        x = np.expand_dims(x, -1)

    if x.shape[-1] != 3:
        x = np.repeat(x, repeats=3, axis=-1)

    if x.dtype != 'uint8':
        x = (255 * x).astype('uint8')

    return x


def make_float_img(x):
    """Converts an image into an float image.

    The image is assumed as numpy array. The function squeezes a channel
    dimension of one and converts uint8 values into float by dividing them
    with 255 first. The method accepts 2 and 3-dimensional inputs.

    # Inputs:
        x: Input image as numpy array of floats in the range [0,1] or unit8
            values. The shape could be 2 or 3-dimensional.

    # Output:
        Input image as numpy array of float values in the range [0,1] with
        shape (rows, cols, 3) or (rows, cols) in case of gray scale images.
    """
    if x.shape[-1] == 1:
        x = np.squeeze(x, -1)

    if x.dtype == 'uint8':
        x = x.astype('uint8') / 255

    return x


def resize_img_stack(stack, shape):
    """Resizes a stack of gray-scale images with opencv's bi-cubic
    interpolation.

    The method takes a `stack` of gray-scale images and resizes all the images
    to the shape `shape`. The resizing method is opencv's resize function with
    bi-cubic interpolation.

    # Inputs:
        stack: Stack of gray-scale images as numpy array with shape
            (rows, cols, number_of_images) or (rows, cols) in case only one
            image should be resized. The data type must be an accepted image
            data type of opencv.
        shape: 2-dimensional list/tuple of integers, specifying the
            target shape of the images as (new_rows, new_cols)

    # Output:
        Resized image stack as numpy array with shape
        (new_rows, new_cols, number_of_images) or (new_rows, new_cols).
    """
    # we flip the shape because opencv assumes (x, y) and we follow the
    # numpy convention (y, x) == (rows, cols)
    new_stack = cv2.resize(stack,
                           dsize=(shape[1], shape[0]),
                           interpolation=cv2.INTER_CUBIC)

    return new_stack
