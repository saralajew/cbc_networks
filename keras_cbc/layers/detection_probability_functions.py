# -*- coding: utf-8 -*-
"""Implementation of detection probability function layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
from keras.utils import conv_utils
from keras import activations

import keras.backend as K
import numpy as np


class CosineSimilarity2D(Layer):
    """2-dimensional sliding operation with the cosine similarity.

    This layer computes the cosine similarity in a 2-dimensional sliding
    operation on a given input. Thereby, it accepts a predefined definition
    of components given by the argument `components` or a second input
    tensor of components.

    The layer can handle multiple versions of one component via `n_replicas`.

    The layer requires `K.image_data_format()` to be 'channels_last'.

    Use the `activation` argument to apply a proper activation to
    convert the similarity in a detection probability function.

    The documentation of the arguments `strides`, `padding`, `dilation_rate`
    and `activation` is copied from the Keras documentation of the `Conv2D`
    layer.

    # Arguments
        n_replicas: An integer specifying the number of replicas for each
            component. This parameter can be used to learn for one component
            multiple versions to handle variations. This could be more
            efficient in terms of parameters than learning multiple
            reasoning concepts. The output number of components is the input
            number of components divided by `n_replicas`. A respective
            handling of the replicas must performed manually afterwards.
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
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (i.e. "linear" activation: `a(x) = x`).
        components: Numpy array,
            specifying predefined components.

    # Input shape
        List of two 4-dimensional tensors with shapes:
        `[(batch, rows, cols, channels),
          (number of components, rows, cols, channels)]`, where the first input
          is the input data feature tensor and the second input is the
          component feature tensor (the kernel)
        or a 4-dimensional tensor with shape:
        `(batch, rows, cols, channels)`
        if `components` is not 'None'.

    # Output shape
        4-dimensional tensor with shape:
        `(batch, new_rows, new_cols, number of components)`
        if `n_replicas` == 1 or a 5-dimensional tensor with shape:
        `(batch, new_rows, new_cols,
          number of components // n_replicas, n_replicas)` otherwise.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_replicas=1,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation='relu',
                 components=None,
                 **kwargs):
        super(CosineSimilarity2D, self).__init__(**kwargs)
        self.n_replicas = n_replicas
        self.rank = 2
        self.strides = conv_utils.normalize_tuple(strides,
                                                  self.rank,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,
                                                        self.rank,
                                                        'dilation_rate')
        self.components = components
        self.activation = activations.get(activation)

        if K.image_data_format() != 'channels_last':
            raise ValueError("The layer requires `K.image_data_format()` to "
                             "be 'channels_last'.")

    def build(self, input_shape):
        # Only one tensor input given (no components input tensor)
        if not (isinstance(input_shape, list) and len(input_shape) > 1):
            # Component initializer should be given
            if self.components is None:
                raise ValueError("Input shape is 1, this requires components "
                                 "to be predefined by the `components` "
                                 "argument.")
            kernel_shape = self.components.shape
        else:
            _, kernel_shape = input_shape

        if kernel_shape[0] % self.n_replicas != 0:
            raise ValueError("The number of components must be a "
                             "multiple of 'n_replicas'.")
        self.built = True

    def call(self, inputs, **kwargs):
        def sqrt(x):
            return K.sqrt(K.maximum(x, K.epsilon()))

        # Both components and input given
        if isinstance(inputs, list) and len(inputs) > 1:
            signals, kernel = inputs
        else:
            signals = inputs
            kernel = self.components.astype(K.floatx())

        # move component_number to channel dimension
        kernel = K.permute_dimensions(kernel, (1, 2, 3, 0))
        # normalize kernel
        normed_kernel = kernel / sqrt(K.sum(K.square(kernel),
                                            (0, 1, 2),
                                            keepdims=True))

        # get norm of signals
        signals_norm = sqrt(K.conv2d(K.square(signals),
                                     np.ones(K.int_shape(kernel)[:3] + (1,),
                                             dtype=K.floatx()),
                                     strides=self.strides,
                                     padding=self.padding,
                                     data_format='channels_last',
                                     dilation_rate=self.dilation_rate))

        diss = K.conv2d(signals,
                        normed_kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format='channels_last',
                        dilation_rate=self.dilation_rate) / signals_norm

        if self.n_replicas != 1:
            shape = K.int_shape(diss)
            diss = K.reshape(diss, (-1, shape[1], shape[2],
                                    shape[3] // self.n_replicas,
                                    self.n_replicas))

        return self.activation(diss)

    def compute_output_shape(self, input_shape):
        # Both components and input given
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape, kernel_shape = input_shape
        else:
            input_shape = input_shape
            kernel_shape = self.components.shape

        filters = kernel_shape[0] // self.n_replicas
        kernel_shape = kernel_shape[1:3]

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                kernel_shape[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        if self.n_replicas != 1:
            return (input_shape[0],) + tuple(new_space) + \
                   (filters, self.n_replicas)
        else:
            return (input_shape[0],) + tuple(new_space) + (filters,)


class EuclideanDistance2D(Layer):
    """2-dimensional sliding operation with the Euclidean distance.

    This layer computes the Euclidean distance in a 2-dimensional sliding
    operation on a given input. Thereby, it accepts a predefined definition
    of components given by the argument `components` or a second input
    tensor of components.

    The layer can handle multiple versions of one component via `n_replicas`.

    It computes the squared version of the Euclidean distance. If the
    non-squared version is requested, then use the activation function to
    apply the sqrt.

    The layer requires `K.image_data_format()` to be 'channels_last'.

    Use the `activation` argument to apply a proper activation to
    convert the distance in a detection probability function.

    The documentation of the arguments `strides`, `padding`, `dilation_rate`
    and `activation` is copied from the Keras documentation of the `Conv2D`
    layer.

    # Arguments
        n_replicas: An integer specifying the number of replicas for each
            component. This parameter can be used to learn for one component
            multiple versions to handle variations. This could be more
            efficient in terms of parameters than learning multiple
            reasoning concepts. The output number of components is the input
            number of components divided by `n_replicas`. A respective
            handling of the replicas must performed manually afterwards.
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
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (i.e. "linear" activation: `a(x) = x`).
        components: Numpy array,
            specifying predefined components.

    # Input shape
        List of two 4-dimensional tensors with shapes:
        `[(batch, rows, cols, channels),
          (number of components, rows, cols, channels)]`, where the first input
          is the input data feature tensor and the second input is the
          component feature tensor (the kernel)
        or a 4-dimensional tensor with shape:
        `(batch, rows, cols, channels)`
        if `components` is not 'None'.

    # Output shape
        4-dimensional tensor with shape:
        `(batch, new_rows, new_cols, number of components)`
        if `n_replicas` == 1 or a 5-dimensional tensor with shape:
        `(batch, new_rows, new_cols,
          number of components // n_replicas, n_replicas)` otherwise.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_replicas=1,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation=lambda x: K.exp(-x),
                 components=None,
                 **kwargs):
        super(EuclideanDistance2D, self).__init__(**kwargs)
        self.n_replicas = n_replicas
        self.rank = 2
        self.strides = conv_utils.normalize_tuple(strides,
                                                  self.rank,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,
                                                        self.rank,
                                                        'dilation_rate')
        self.components = components
        self.activation = activations.get(activation)

        if K.image_data_format() != 'channels_last':
            raise ValueError("The layer requires `K.image_data_format()` to "
                             "be 'channels_last'.")

    def build(self, input_shape):
        # Only one tensor input given (no components input tensor)
        if not (isinstance(input_shape, list) and len(input_shape) > 1):
            # Component initializer should be given
            if self.components is None:
                raise ValueError("Input shape is 1, this requires components "
                                 "to be predefined by the `components` "
                                 "argument.")
            kernel_shape = self.components.shape
        else:
            _, kernel_shape = input_shape

        if kernel_shape[0] % self.n_replicas != 0:
            raise ValueError("The number of components must be a "
                             "multiple of 'n_replicas'.")
        self.built = True

    def call(self, inputs, **kwargs):
        # Both components and input given
        if isinstance(inputs, list) and len(inputs) > 1:
            signals, kernel = inputs
        else:
            signals = inputs
            kernel = self.components.astype(K.floatx())

        # move component_number to channel dimension
        kernel = K.permute_dimensions(kernel, (1, 2, 3, 0))
        # normalize kernel
        kernel_norm = K.sum(K.square(kernel), (0, 1, 2), keepdims=True)

        # get norm of signals
        signals_norm = K.conv2d(K.square(signals),
                                K.ones_like(kernel),
                                strides=self.strides,
                                padding=self.padding,
                                data_format='channels_last',
                                dilation_rate=self.dilation_rate)

        diss = kernel_norm \
               - 2 * K.conv2d(signals,
                              kernel,
                              strides=self.strides,
                              padding=self.padding,
                              data_format='channels_last',
                              dilation_rate=self.dilation_rate) \
               + signals_norm

        if self.n_replicas != 1:
            shape = K.int_shape(diss)
            diss = K.reshape(diss, (-1, shape[1], shape[2],
                                    shape[3] // self.n_replicas,
                                    self.n_replicas))

        return self.activation(diss)

    def compute_output_shape(self, input_shape):
        # Both components and input given
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape, kernel_shape = input_shape
        else:
            input_shape = input_shape
            kernel_shape = self.components.shape

        filters = kernel_shape[0] // self.n_replicas
        kernel_shape = kernel_shape[1:3]

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                kernel_shape[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        if self.n_replicas != 1:
            return (input_shape[0],) + tuple(new_space) + \
                   (filters, self.n_replicas)
        else:
            return (input_shape[0],) + tuple(new_space) + (filters,)
