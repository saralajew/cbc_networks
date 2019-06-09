# -*- coding: utf-8 -*-
"""Input independent spatial reasoning visualizations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

from keras_cbc.visualizations.utils import make_uint8_img, pixel_counts, \
    patch_to_img, resize_img_stack

import numpy as np


def plot_optimal_reasoning_heatmaps(positive_effective_reasoning,
                                    negative_effective_reasoning,
                                    pixel_probabilities,
                                    resized_reasoning_shape,
                                    path):
    """Plot of the optimal reasoning heatmaps.

    This function plots the optimal reasoning heatmaps. Hence, it visualizes
    positive/negative agreement. The function plots the visualization for
    all the replicas.

    # Arguments:
        positive_effective_reasoning: Numpy array of the positive effective
            reasoning probabilities. The shape is
            (kernel_shape[0], kernel_shape[1],
            n_components, n_classes, n_replicas).
        negative_effective_reasoning: Numpy array of the negative effective
            reasoning probabilities. The shape is
            (kernel_shape[0], kernel_shape[1],
            n_components, n_classes, n_replicas).
        detection_probability: Numpy array of the detection probabilities.
            The shape is (batch, rows, cols, n_components).
        pixel_probabilities: Numpy array of pixel probabilities. The shape is
            (kernel_shape[0], kernel_shape[1], 1, n_classes, n_replicas).
        resized_reasoning_shape: A tuple/list of 2 integers, specifying the
            receptive field of the reasoning process in the input space.
        path: String, specifying the path where the images have to be
            stored.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    n_classes = positive_effective_reasoning.shape[3]
    n_replicas = positive_effective_reasoning.shape[4]

    for i in range(n_classes):
        for j in range(n_replicas):
            # positive and negative agreement
            for k, R in enumerate([positive_effective_reasoning,
                                   negative_effective_reasoning]):
                reasoning = R[:, :, :, i, j]

                heatmap = resize_img_stack(reasoning, resized_reasoning_shape)
                heatmap = np.sum(heatmap, -1)
                heatmap = np.clip(heatmap, 0, 1)

                pixel_heatmap = resize_img_stack(
                    pixel_probabilities[:, :, 0, i, j],
                    resized_reasoning_shape)
                pixel_heatmap = np.clip(pixel_heatmap, 0, 1)

                heatmap = heatmap * pixel_heatmap
                overlay = make_uint8_img(heatmap)
                heatmap_img = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

                if k == 0:
                    prefix = '_pos_'
                else:
                    prefix = '_neg_'
                cv2.imwrite(path +
                            'class_' + str(i) +
                            prefix +
                            'replica_' + str(j) + '.png', heatmap_img)


def plot_optimal_reconstruction(components,
                                positive_effective_reasoning,
                                negative_effective_reasoning,
                                pixel_probabilities,
                                resized_kernel_shape,
                                path):
    """Plot of the optimal reasoning reconstructions.

    This function plots the optimal reasoning reconstruction. Hence,
    it visualizes positive/negative agreement. The function plots the
    visualization for all the replicas.
    The function is only defined for gray-scale components and assumes
    components defined over [0,1].
    Multiple components (component replicas) are not supported.

    # Arguments:
        components: Numpy stack of components of shape
            (n_components, rows, cols, 1) defined over the range [0,1].
        positive_effective_reasoning: Numpy array of the positive effective
            reasoning probabilities. The shape is
            (kernel_shape[0], kernel_shape[1],
            n_components, n_classes, n_replicas).
        negative_effective_reasoning: Numpy array of the negative effective
            reasoning probabilities. The shape is
            (kernel_shape[0], kernel_shape[1],
            n_components, n_classes, n_replicas).
        pixel_probabilities: Numpy array of pixel probabilities. The shape is
            (kernel_shape[0], kernel_shape[1], 1, n_classes, n_replicas).
        resized_kernel_shape: A tuple/list of 2 integers, specifying the
            resized spatial reasoning kernel shape which is used for the
            reconstructions.
        path: String, specifying the path where the images have to be
            stored.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    if components.shape[0] != positive_effective_reasoning.shape[2]:
        raise ValueError("The function doesn't support multiple component "
                         "versions. Use component n_replicas=1.")

    component_shape = components.shape[1:]
    n_classes = positive_effective_reasoning.shape[3]
    n_replicas = positive_effective_reasoning.shape[4]
    reconstruction_shape = (resized_kernel_shape[0] + component_shape[0] - 1,
                            resized_kernel_shape[1] + component_shape[1] - 1)

    if components.shape[-1] != 1:
        raise ValueError("Only gray-scale components are supported.")
    components = np.transpose(np.squeeze(components, -1), (1, 2, 0))

    normalizer = pixel_counts(resized_kernel_shape, component_shape)

    for i in range(n_classes):
        for j in range(n_replicas):
            # positive and negative images
            for k, R in enumerate([positive_effective_reasoning,
                                   negative_effective_reasoning]):
                reasoning = R[:, :, :, i, j]

                # extend to size before pooling; e.g. the cube is 9x9 --> 18x18
                resized_reasoning = resize_img_stack(reasoning,
                                                     resized_kernel_shape)
                resized_reasoning = np.clip(resized_reasoning, 0., 1.)

                # check that the sum is not greater than one! It must hold.
                for m in range(resized_kernel_shape[0]):
                    for n in range(resized_kernel_shape[1]):
                        sum_prob = np.sum(resized_reasoning[m, n])
                        if sum_prob > 1:
                            resized_reasoning[m, n] = \
                                resized_reasoning[m, n] / sum_prob

                # patch components into image and normalize
                reconstructed_img = patch_to_img(resized_reasoning, components)
                reconstructed_img /= normalizer

                # create pixel heatmap
                pixel_heatmap = resize_img_stack(
                    pixel_probabilities[:, :, 0, i, j],
                    reconstruction_shape)
                pixel_heatmap = np.clip(pixel_heatmap, 0, 1)

                # overlay pixel heatmap and create image
                reconstructed_img = pixel_heatmap * reconstructed_img
                reconstructed_img = cv2.cvtColor(
                    make_uint8_img(reconstructed_img), cv2.COLOR_RGB2BGR)

                if k == 0:
                    prefix = '_pos_'
                else:
                    prefix = '_neg_'
                cv2.imwrite(path +
                            'class_' + str(i) +
                            prefix +
                            'replica_' + str(j) + '.png', reconstructed_img)
