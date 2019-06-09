# -*- coding: utf-8 -*-
"""Visualization methods for learned components.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

from keras_cbc.visualizations.utils import make_uint8_img, resize_img_stack

import numpy as np


def plot_components(components, path):
    """Plot of the components as images.

    The function stores the components as images. For that the components
    are assumed to be trained in the input space of images which is defined
    over the range [0,1]. The naming of stored components is
    '<component_number>.png'.

    # Inputs:
        components: Numpy array of float values in the range [0,1],
            specifying the components. The shape is
            (number_of_components, rows, cols).
        path: String, specifying the path where the images have to be
            stored.

    # Output:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    for i, img in enumerate(components):
        cv2.imwrite(path + str(i) + '.png',
                    cv2.cvtColor(make_uint8_img(img), cv2.COLOR_RGB2BGR))


def plot_simple_reasoning(reasoning, path, component_probabilities=None):
    """Plot of a simple reasoning process.

    Simple reasoning refers to a non-spatial reasoning process. Multiple
    reasoning is supported. The function visualizes the reasoning process
    and stores the images in `path`. If component probabilities are
    provided, then the method scales the reasoning probabilities by the
    component probabilities respectively. Moreover, the reasoning
    probabilities are visualized and stored in an own image.

    # Inputs:
        reasoning: Numpy array of float values in the range [0,1] which are
            the encoded reasoning weights of 'Reasoning' layer. The shape is
            (2, n_replicas, n_components, n_classes)
        path: String, specifying the path where the images have to be
            stored.
        component_probabilities: Numpy array of float values, which are
            specifying the encoded component probabilities of a 'Reasoning'
            layer. The shape is (1, n_components, 1).

    # Output:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    if reasoning.ndim != 4:
        raise ValueError("Only simple reasoning is support and, hence, "
                         "the reasoning weights are assumed to be "
                         "4-dimensional. The method doesn't support spatial "
                         "reasoning.")

    # decoding reasoning
    positive = reasoning[0]
    negative = (1 - positive) * reasoning[1]
    indefinite = 1 - positive - negative

    # stack reasoning matrices
    reasoning_matrices = np.stack([positive, indefinite, negative],
                                  axis=0)

    n_replicas = positive.shape[0]
    n_components = positive.shape[1]
    n_classes = positive.shape[2]

    if component_probabilities is not None:
        # decoding component probabilities
        component_probabilities = component_probabilities \
                                  - np.max(component_probabilities)
        component_probabilities = np.exp(component_probabilities)
        component_probabilities = component_probabilities / \
                                  np.sum(component_probabilities,
                                         axis=1,
                                         keepdims=True)
        # plot probabilities
        img = make_uint8_img(component_probabilities[:, :, 0])
        img = cv2.resize(img,
                         dsize=(10 * n_components, 10),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path + 'component_probabilities.png', img)

    else:
        # default value if no component probabilities are given
        component_probabilities = 1

    # scaling of the reasoning matrices with the component probabilities
    reasoning_matrices = reasoning_matrices * component_probabilities

    # plot reasoning matrices
    img = np.zeros((32, 10 * n_components + 2, 3), dtype='uint8')
    for i in range(n_classes):
        for j in range(n_replicas):
            reasoning_img = make_uint8_img(reasoning_matrices[:, j, :, i])
            reasoning_img = cv2.resize(reasoning_img,
                                       dsize=(10 * n_components, 30),
                                       interpolation=cv2.INTER_NEAREST)
            # draw black borders
            img[1:-1, 1:-1] = reasoning_img
            img[0::10] = 0
            img[1::10] = 0
            img[:, 0::10] = 0
            img[:, 1::10] = 0

            cv2.imwrite(path +
                        'class_' + str(i) +
                        '_replica_' + str(j) + '.png',
                        img)


def plot_components_heatmaps(probabilities, x_test, path):
    """Plots the component heatmaps.

    The component heatmaps are the probabilities after the evaluation of
    the detection probabilities. The heatmaps visualizing the response to
    the component at the respective position. Moreover, this visualization
    depends on an input and, hence, the input is highlighted in the background.
    probabilities are resized to the size of the input sample.

    The color coding for the heatmaps is opencv's 'JET' colormap. Thus,
    a response (probability) of 0 is mapped to deep blue and response of 1
    is mapped to hot red. Intermediate values are interpolated appropriately.

    # Inputs:
        probabilities: Numpy array of detection probabilities for each sample
            in `x_test`. The shape is (batch, rows, cols, n_components).
        x_test: Numpy array of test images of shape
            (batch, v, h, channels).
        path: String, specifying the path where the images have to be
            stored.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    n_components = probabilities.shape[-1]

    for i in range(x_test.shape[0]):
        for j in range(n_components):
            heatmap = resize_img_stack(probabilities[i, :, :, j],
                                       x_test.shape[1:3])
            # clip to catch interpolations which are outside [0,1]
            heatmap = np.clip(heatmap, 0, 1)
            heatmap = make_uint8_img(heatmap)
            heatmap_img = cv2.applyColorMap(heatmap[:, :, 0], cv2.COLORMAP_JET)

            background_img = make_uint8_img(x_test[i])

            img = cv2.addWeighted(heatmap_img, 0.5, background_img, 0.5, 0)
            cv2.imwrite(path +
                        '/sample_' + str(i) +
                        '_comp_' + str(j) + '.png', img)
