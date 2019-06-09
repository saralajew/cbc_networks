# -*- coding: utf-8 -*-
"""Implementation of reasoning layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer, InputSpec, conv_utils
from keras import regularizers, initializers, constraints

import keras.backend as K
import numpy as np


class Reasoning(Layer):
    """Simple reasoning over one detection possibility vector.

    This layer computes the class-wise hypothesis probabilities based on a
    reasoning process over one detection possibility vector. The output is
    the class hypothesis possibility vector. Hence, this layer is in favor
    similar to a traditional dense layer.

    The `reasoning_intializer` and `reasoning_regularizer` is applied on the
    encoded reasoning probabilities. Moreover, the component probabilities
    are trained over IR and squashed by softmax to a probability vector.
    Hence, the intializer/regualrizer/constraint applies over IR.

    # Arguments
        n_classes: Integer, specifying the number of classes.
        n_replicas: Integer, specifying the number of trainable reasoning
            processes for each class. A value greater than 1 realizes
            multiple reasoning. A respective handling of the replicas must
            performed manually afterwards.
        reasoning_initializer: Initializer for the encoded
            reasoning probabilities  which is interpretable by Keras
            `initializers.get()` routine.
        reasoning_regularizer: Regularizer for the encoded
            reasoning probabilities  which is interpretable by Keras
            `regularizers.get()` routine.
        use_component_probabilities: Boolean, specifying if the reasoning
            process has trainable component probabilities. If false,
            the model assumes a probability of 1/number_of_components for
            all components.
        component_probabilities_initializer: Initializer for the component
            probabilities  which is interpretable by Keras
            `initializers.get()` routine.
        component_probabilities_regularizer: Regularizer for the component
            probabilities  which is interpretable by Keras
            `regularizers.get()` routine.
        component_probabilities_constraint: Constraint for the component
            probabilities  which is interpretable by Keras
            `constraints.get()` routine.

    # Input shape
        2-dimensional tensor with shape:
        `(batch, number of components)`.
        This tensor is the detection possibility vector for each batch.

    # Output shape
        2-dimensional tensor with shape
        `(batch, n_classes)` if `n_replicas` == 1 or
        3-dimensional tensor with shape:
        `(batch, n_classes, n_relpicas)` otherwise.
        This tensor is the class hypothesis possibility vector for each batch.
    """

    def __init__(self,
                 n_classes,
                 n_replicas=1,
                 reasoning_initializer='zeros',
                 reasoning_regularizer=None,
                 use_component_probabilities=False,
                 component_probabilities_initializer='zeros',
                 component_probabilities_regularizer=None,
                 component_probabilities_constraint=None,
                 **kwargs):
        super(Reasoning, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_replicas = n_replicas

        self.reasoning_initializer = initializers.get(reasoning_initializer)
        self.reasoning_regularizer = regularizers.get(reasoning_regularizer)

        self.use_component_probabilities = use_component_probabilities
        self.component_probabilities_initializer = initializers.get(
            component_probabilities_initializer)
        self.component_probabilities_regularizer = regularizers.get(
            component_probabilities_regularizer)
        self.component_probabilities_constraint = constraints.get(
            component_probabilities_constraint)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=(None,) + tuple(input_shape[1:]))

        # encoded trainable tensors
        self.reasoning_probabilities = self.add_weight(
            shape=(2,
                   self.n_replicas,
                   input_shape[-1],
                   self.n_classes),
            initializer=self.reasoning_initializer,
            regularizer=self.reasoning_regularizer,
            constraint=lambda x: K.clip(x, 0., 1.),
            name='reasoning_probabilities')

        if self.use_component_probabilities:
            self.component_probabilities = self.add_weight(
                shape=(1, input_shape[-1], 1),
                initializer=self.component_probabilities_initializer,
                regularizer=self.component_probabilities_regularizer,
                constraint=self.component_probabilities_constraint,
                name='component_probabilities')

        self.built = True

    def call(self, inputs, **kwargs):
        # decode the reasoning probabilities
        positive_kernel = self.reasoning_probabilities[0]
        negative_kernel = (1 - positive_kernel) * \
                          self.reasoning_probabilities[1]

        if self.use_component_probabilities:
            # squash component probabilities
            components_probabilities = softmax(self.component_probabilities)

            positive_kernel = positive_kernel * components_probabilities
            negative_kernel = negative_kernel * components_probabilities

        # stabilize the division with a small epsilon
        probs = (K.dot(inputs, (positive_kernel - negative_kernel)) \
                 + K.sum(negative_kernel, 1)) \
                / (K.sum(positive_kernel + negative_kernel, 1) + K.epsilon())

        # squeeze replica dimension if one.
        if self.n_replicas == 1:
            probs = K.squeeze(probs, axis=1)
        else:
            probs = K.permute_dimensions(probs, (0, 2, 1))

        return probs

    def compute_output_shape(self, input_shape):
        if self.n_replicas != 1:
            return (None, self.n_classes, self.n_replicas)
        else:
            return (None, self.n_classes)


class Reasoning2D(Layer):
    """Spatial reasoning over a detection possibility stack.

    This layer computes the class-wise hypothesis probabilities based on
    spatial reasoning. The output is a class hypothesis possibility stack.
    This implementation is the extension of the simple reasoning process to
    a sliding operation.

    The `reasoning_intializer` and `reasoning_regularizer` is applied on the
    encoded reasoning probabilities. Moreover, the component probabilities
    are trained over IR and squashed by softmax to a probability vector.
    Hence, the intializer/regualrizer/constraint applies over IR. The same
    holds for the pixel probabilities.

    The documentation of the arguments `strides`, `padding`, `dilation_rate`
    and `activation` is copied from the Keras documentation of the `Conv2D`
    layer.

    # Arguments
        n_classes: Integer, specifying the number of classes.
        n_replicas: Integer, specifying the number of trainable reasoning
            processes for each class. A value greater than 1 realizes
            multiple reasoning.  A respective handling of the replicas must
            performed manually afterwards.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D spatial reasoning stack.
            Can be a single integer to specify the same value for
            all spatial dimensions. If 'None', then the kernel_size is
            automatically defined as the spatial input dimension size.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1, as described
            [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        reasoning_initializer: Initializer for the encoded
            reasoning probabilities  which is interpretable by Keras
            `initializers.get()` routine.
        reasoning_regularizer: Regularizer for the encoded
            reasoning probabilities  which is interpretable by Keras
            `regularizers.get()` routine.
        use_component_probabilities: Boolean, specifying if the reasoning
            process has trainable component probabilities. If false,
            the model assumes a probability of 1/number_of_components for
            all components.
        component_probabilities_initializer: Initializer for the component
            probabilities  which is interpretable by Keras
            `initializers.get()` routine.
        component_probabilities_regularizer: Regularizer for the component
            probabilities  which is interpretable by Keras
            `regularizers.get()` routine.
        component_probabilities_constraint: Constraint for the component
            probabilities  which is interpretable by Keras
            `constraints.get()` routine.
        use_pixel_probabilities: Boolean, specifying if the reasoning
            process has trainable pixel probabilities. If false,
            the model assumes a probability of 1/kernel_size.
        pixel_probabilities_initializer: Initializer for the component
            probabilities  which is interpretable by Keras
            `initializers.get()` routine.
        pixel_probabilities_regularizer: Regularizer for the component
            probabilities  which is interpretable by Keras
            `regularizers.get()` routine.
        pixel_probabilities_constraint: Constraint for the component
            probabilities  which is interpretable by Keras
            `constraints.get()` routine.

    # Input shape
        4-dimensional tensor with shape:
        `(batch, rows, cols, number of components)`.
        This tensor is the detection possibility stack for each batch.

    # Output shape
        4-dimensional tensor with shape:
        `(batch, new_rows, new_cols, n_classes)` if `n_replicas` == 1 or
        5-dimensional tensor with shape:
        `(batch, new_rows, new_cols, n_classes, n_replicas)` otherwise.
        `rows` and `cols` values might have changed due to padding.
        This tensor is the class hypothesis possibility stack for each batch.
    """

    def __init__(self,
                 n_classes,
                 n_replicas=1,
                 kernel_size=None,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 reasoning_initializer='zeros',
                 reasoning_regularizer=None,
                 use_component_probabilities=False,
                 component_probabilities_initializer='zeros',
                 component_probabilities_regularizer=None,
                 component_probabilities_constraint=None,
                 use_pixel_probabilities=False,
                 pixel_probabilities_initializer='zeros',
                 pixel_probabilities_regularizer=None,
                 pixel_probabilities_constraint=None,
                 **kwargs):
        super(Reasoning2D, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.n_replicas = n_replicas

        self.rank = 2
        if kernel_size is not None:
            self.kernel_size = conv_utils.normalize_tuple(kernel_size,
                                                          self.rank,
                                                          'kernel_size')
        else:
            self.kernel_size = None
        self.strides = conv_utils.normalize_tuple(strides,
                                                  self.rank,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,
                                                        self.rank,
                                                        'dilation_rate')

        self.reasoning_initializer = initializers.get(reasoning_initializer)
        self.reasoning_regularizer = regularizers.get(reasoning_regularizer)

        self.use_component_probabilities = use_component_probabilities
        self.component_probabilities_initializer = initializers.get(
            component_probabilities_initializer)
        self.component_probabilities_regularizer = regularizers.get(
            component_probabilities_regularizer)
        self.component_probabilities_constraint = constraints.get(
            component_probabilities_constraint)

        self.use_pixel_probabilities = use_pixel_probabilities
        self.pixel_probabilities_initializer = initializers.get(
            pixel_probabilities_initializer)
        self.pixel_probabilities_regularizer = regularizers.get(
            pixel_probabilities_regularizer)
        self.pixel_probabilities_constraint = constraints.get(
            pixel_probabilities_constraint)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=(None,) + tuple(input_shape[1:]))

        # define kernel_size as full-image if not provided
        if self.kernel_size is None:
            self.kernel_size = input_shape[1:3]

        kernel_shape = (2,) \
                       + self.kernel_size \
                       + (input_shape[-1], self.n_classes * self.n_replicas)

        # encoded trainable tensors
        self.reasoning_probabilities = self.add_weight(
            shape=kernel_shape,
            initializer=self.reasoning_initializer,
            regularizer=self.reasoning_regularizer,
            constraint=lambda x: K.clip(x, 0., 1.),
            name='reasoning_probabilities')

        if self.use_pixel_probabilities:
            self.pixel_probabilities = self.add_weight(
                shape=self.kernel_size + (1, self.n_classes * self.n_replicas),
                initializer=self.pixel_probabilities_initializer,
                regularizer=self.pixel_probabilities_regularizer,
                constraint=self.pixel_probabilities_constraint,
                name='pixel_probabilities')

        if self.use_component_probabilities:
            self.component_probabilities = self.add_weight(
                shape=(1, 1, input_shape[-1], 1),
                initializer=self.component_probabilities_initializer,
                regularizer=self.component_probabilities_regularizer,
                constraint=self.component_probabilities_constraint,
                name='component_probabilities')

        self.built = True

    def call(self, inputs, **kwargs):
        # decode the reasoning probabilities
        positive_kernel = self.reasoning_probabilities[0]
        negative_kernel = (1 - positive_kernel) * \
                          self.reasoning_probabilities[1]

        if self.use_component_probabilities:
            # squash component probabilities
            components_probabilities = softmax(self.component_probabilities,
                                               axis=2)

            positive_kernel = positive_kernel * components_probabilities
            negative_kernel = negative_kernel * components_probabilities

        # get normalization tensor
        # stabilize the division with a small epsilon
        normalization = K.sum(positive_kernel + negative_kernel,
                              axis=2,
                              keepdims=True) + K.epsilon()

        # get sliding kernel and bias
        if self.use_pixel_probabilities:
            pixel_probabilities = softmax(self.pixel_probabilities,
                                          axis=(0, 1))
            # scale kernel with priors
            kernel = (positive_kernel - negative_kernel) / normalization \
                     * pixel_probabilities
            bias = K.sum(negative_kernel / normalization
                         * pixel_probabilities,
                         axis=(0, 1, 2),
                         keepdims=True)
        else:
            kernel = (positive_kernel - negative_kernel) / normalization
            bias = K.sum(negative_kernel / normalization,
                         axis=(0, 1, 2),
                         keepdims=True)

        # compute probabilities by a sliding operation
        probs = K.conv2d(inputs, kernel,
                         strides=self.strides,
                         padding=self.padding,
                         data_format='channels_last',
                         dilation_rate=self.dilation_rate) + bias

        if not self.use_pixel_probabilities:
            # divide by number of kernel_size
            probs = probs / np.prod(self.kernel_size)

        # reshape to m x n x #classes x #replicas
        probs = K.reshape(probs,
                          (-1,) + K.int_shape(probs)[1:3]
                          + (self.n_classes, self.n_replicas))

        # squeeze replica dimension if one.
        if self.n_replicas == 1:
            probs = K.squeeze(probs, axis=-1)

        return probs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        shape = list((input_shape[0],) + tuple(new_space) + (self.n_classes,))

        if self.n_replicas != 1:
            shape = shape + [self.n_replicas]

        return tuple(shape)


def softmax(tensors, axis=-1):
    """Implementation of softmax with maximum stabilization and multiple
    axis support.

    # Arguments
        tensors: Input tensor.
        axis: An integer or tuple/list of integers, specifying the
            axis for the normalization

    # Input shape
        tensor with arbitrary shape

    # Output shape
        tensor with the same shape as the input tensor
    """
    with K.name_scope('softmax'):
        tensors = tensors - K.max(tensors, axis=axis, keepdims=True)
        exp = K.exp(tensors)
        return exp / K.sum(exp, axis=axis, keepdims=True)
