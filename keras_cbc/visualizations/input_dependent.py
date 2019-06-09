# -*- coding: utf-8 -*-
"""Input dependent spatial reasoning visualizations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

from keras_cbc.visualizations.utils import make_uint8_img, pixel_counts, \
    patch_to_img, resize_img_stack

import numpy as np


def plot_input_reasoning_heatmaps(positive_effective_reasoning,
                                  negative_effective_reasoning,
                                  detection_probability,
                                  pixel_probabilities,
                                  wta_idx,
                                  resized_reasoning_shape,
                                  x_test,
                                  path):
    """Plot of the reasoning heatmaps regarding inputs.

    This function plots the reasoning heatmaps regarding given inputs.
    Hence, it visualizes positive/negative agreement/disagreement. The
    heatmaps are overlaid with the corresponding input image.
    The function can handle spatial reasoning processes where the spatial
    reasoning shape is lower than the detection probability shape. In this
    case it uses the best matching position of the reasoning process from
    the wta_idx and back propagates the spatial reasoning shape to the input
    shape. Note that in this case the resulting reasoning shape could be
    lower than the spatial input shape. The gap is padded with zero values.
    Moreover, it handles multiple reasoning by highlighting the best
    matching replica.

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
        wta_idx: List with nested numpy array containing the position at
            which the spatial reasoning reached the highest probability
            using the best matching replica. The list contains the following
            elements:
                [0]: Numpy array of the best matching vertical position of a
                    certain sample and class. The shape is (batch, n_classes).
                [1]: Numpy array of the best matching horizontal position of a
                    certain sample and class. The shape is (batch, n_classes).
                [2]: Numpy array of the best matching replica index of a
                    certain sample and class. The shape is (batch, n_classes).
        resized_reasoning_shape: A tuple/list of 2 integers, specifying the
            receptive field of the reasoning process in the input space.
        x_test: Numpy array of test images of shape
            (batch, v, h, channels).
        path: String, specifying the path where the images have to be
            stored.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    kernel_shape = positive_effective_reasoning.shape[:2]
    input_shape = x_test.shape[1:]
    n_classes = positive_effective_reasoning.shape[3]

    for i in range(x_test.shape[0]):
        for j in range(n_classes):
            # convolution position with the highest score
            v0, v1 = wta_idx[0][i, j], wta_idx[0][i, j] + kernel_shape[0]
            h0, h1 = wta_idx[1][i, j], wta_idx[1][i, j] + kernel_shape[1]

            # positive and negative agreement/disagreement
            for k, R in enumerate([positive_effective_reasoning,
                                   negative_effective_reasoning,
                                   positive_effective_reasoning,
                                   negative_effective_reasoning]):
                # best replica
                best_replica = wta_idx[2][i, j]

                # select the detection event
                if k in (1, 2):
                    detection = 1 - detection_probability[i, v0:v1, h0:h1]
                else:
                    detection = detection_probability[i, v0:v1, h0:h1]

                # select best matching reasoning process
                reasoning = R[:, :, :, j, best_replica]

                reasoning = resize_img_stack(reasoning,
                                             resized_reasoning_shape)
                reasoning = np.clip(reasoning, 0, 1)
                detection = resize_img_stack(detection,
                                             resized_reasoning_shape)
                detection = np.clip(detection, 0, 1)
                pixel_heatmap = resize_img_stack(
                    pixel_probabilities[:, :, 0, j, best_replica],
                    resized_reasoning_shape)
                pixel_heatmap = np.clip(pixel_heatmap, 0, 1)

                # create heatmap
                heatmap = np.sum(reasoning * detection, -1)
                heatmap = np.clip(heatmap, 0, 1)
                heatmap = heatmap * pixel_heatmap

                # add heatmap in overlay where borders are zero padded
                overlay = np.zeros(input_shape[:2])
                overlay[v0:v0 + resized_reasoning_shape[0],
                h0:h0 + resized_reasoning_shape[1]] = heatmap

                overlay = make_uint8_img(overlay)
                heatmap_img = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

                background_img = \
                    cv2.cvtColor(make_uint8_img(x_test[i]), cv2.COLOR_RGB2BGR)
                fin = cv2.addWeighted(heatmap_img, 0.66,
                                      background_img, 0.34, 0)

                if k == 0:
                    prefix = '_pos_'
                elif k == 1:
                    prefix = '_neg_'
                elif k == 2:
                    prefix = '_pos_not_'
                else:
                    prefix = '_neg_not_'
                cv2.imwrite(path +
                            'sample_' + str(i) +
                            prefix +
                            'class_' + str(j) + '.png', fin)


def plot_input_reconstruction(components,
                              positive_effective_reasoning,
                              negative_effective_reasoning,
                              detection_probability,
                              pixel_probabilities,
                              wta_idx,
                              resized_kernel_shape,
                              x_test,
                              path):
    """Plot of the reasoning reconstructions regarding inputs.

    This function constructs the reasoning reconstructions regarding given
    inputs. Hence, it visualizes positive/negative agreement/disagreement.
    The function can handle spatial reasoning processes where the spatial
    reasoning shape is lower than the detection probability shape. In this
    case it uses the best matching position of the reasoning process from
    the wta_idx and back propagates the spatial reasoning shape to the input
    shape. Note that in this case the resulting reasoning shape could be
    lower than the spatial input shape. The gap is padded with zero values.
    Moreover, it handles multiple reasoning by highlighting the best
    matching replica.
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
        detection_probability: Numpy array of the detection probabilities.
            The shape is (batch, rows, cols, n_components).
        pixel_probabilities: Numpy array of pixel probabilities. The shape is
            (kernel_shape[0], kernel_shape[1], 1, n_classes, n_replicas).
        wta_idx: List with nested numpy array containing the position at
            which the spatial reasoning reached the highest probability
            using the best matching replica. The list contains the following
            elements:
                [0]: Numpy array of the best matching vertical position of a
                    certain sample and class. The shape is (batch, n_classes).
                [1]: Numpy array of the best matching horizontal position of a
                    certain sample and class. The shape is (batch, n_classes).
                [2]: Numpy array of the best matching replica index of a
                    certain sample and class. The shape is (batch, n_classes).
        resized_kernel_shape: A tuple/list of 2 integers, specifying the
            resized spatial reasoning kernel shape which is used for the
            reconstructions.
        x_test: Numpy array of test images of shape
            (batch, v, h, channels).
        path: String, specifying the path where the images have to be
            stored.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/'

    if components.shape[0] != detection_probability.shape[-1]:
        raise ValueError("The function doesn't support multiple component "
                         "versions. Use component n_replicas=1.")

    component_shape = components.shape[1:]
    kernel_shape = positive_effective_reasoning.shape[:2]
    n_classes = positive_effective_reasoning.shape[3]
    reconstruction_shape = (resized_kernel_shape[0] + component_shape[0] - 1,
                            resized_kernel_shape[1] + component_shape[1] - 1)

    if components.shape[-1] != 1:
        raise ValueError("Only gray-scale components are supported.")
    components = np.transpose(np.squeeze(components, -1), (1, 2, 0))

    normalizer = pixel_counts(resized_kernel_shape, component_shape)

    for i in range(x_test.shape[0]):
        for j in range(n_classes):
            # convolution position with the highest score
            v0, v1 = wta_idx[0][i, j], wta_idx[0][i, j] + kernel_shape[0]
            h0, h1 = wta_idx[1][i, j], wta_idx[1][i, j] + kernel_shape[1]

            # positive and negative agreement/disagreement
            for k, R in enumerate([positive_effective_reasoning,
                                   negative_effective_reasoning,
                                   positive_effective_reasoning,
                                   negative_effective_reasoning]):
                # best replica
                best_replica = wta_idx[2][i, j]

                # select the detection event
                if k in (1, 2):
                    detection = 1 - detection_probability[i, v0:v1, h0:h1]
                else:
                    detection = detection_probability[i, v0:v1, h0:h1]

                # select best matching reasoning process
                reasoning = R[:, :, :, j, best_replica]

                # extend to size before pooling; e.g. the cube is 9x9 --> 18x18
                reasoning = resize_img_stack(reasoning, resized_kernel_shape)
                reasoning = np.clip(reasoning, 0, 1)
                detection = resize_img_stack(detection, resized_kernel_shape)
                detection = np.clip(detection, 0, 1)

                # compute the patch scaling factors
                resized_reasoning = reasoning * detection

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
                    pixel_probabilities[:, :, 0, j, best_replica],
                    reconstruction_shape)
                pixel_heatmap = np.clip(pixel_heatmap, 0, 1)

                # overlay pixel heatmap and create image
                reconstructed_img = pixel_heatmap * reconstructed_img
                reconstructed_img = cv2.cvtColor(
                    make_uint8_img(reconstructed_img), cv2.COLOR_RGB2BGR)

                if k == 0:
                    prefix = '_pos_'
                elif k == 1:
                    prefix = '_neg_'
                elif k == 2:
                    prefix = '_pos_not_'
                else:
                    prefix = '_neg_not_'
                cv2.imwrite(path +
                            'sample_' + str(i) +
                            prefix +
                            'class_' + str(j) + '.png', reconstructed_img)
