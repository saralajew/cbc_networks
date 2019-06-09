# -*- coding: utf-8 -*-
"""Layers to initialize components.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer, Input
from keras import regularizers, initializers, constraints

import keras.backend as K


class ConstantInput(object):
    """Constant input stream defined by a value.

    This class creates a constant input stream of values which is used as input
    stream to the Siamese network path of the components. If the object is
    called it returns the constant input stream. This class can be used to add
    pre-defined non-trainable components to the model.

    # Arguments
        variable: An input which is Keras-variable convertible, e.g. a numpy
            array. The variable is used as non-trainable constant Keras input
            stream.
        name: A string. It's the prefix of the Keras variable and input layer
            name.

    # Output shape
        Tensor with the shape and value of the given `variable`.
    """

    def __init__(self, variable, name=''):
        self.variable = variable
        if name != '':
            self.name = name + '_'
        self.variable = K.variable(self.variable,
                                   name=self.name + 'constant_input_variable')
        self.input = Input(tensor=self.variable,
                           name=self.name + 'constant_input')

    def __call__(self, *args, **kwargs):
        return self.input


class AddComponents(Layer):
    """Adds components.

    This layer is used to provide (non-)trainable components to the model. It
    adds a bias tensor (the components) to the input. The addition supports
    broadcasting.

    Together with the ConstantInput class it provides the module to initialize
    trainable components in the input space.

    # Arguments
        shape: An tuple/list of at least 2 integers, specifying the shape of
            the components. The first element is reserved for the number of
            components and, moreover, the following specifying the shape of a
            component.
        initializer: Initializer for the `components` which is interpretable by
            Keras `initializers.get()` routine.
        regularizer: Regularizer for the `components` which is interpretable by
            Keras `regularizers.get()` routine.
        constraint: Constraint for the `components` which is interpretable by
            Keras `constraints.get()` routine.

    # Input shape
        Tensor with a broadcasting compatible shape to `shape`.

    # Output shape
        Tensor with the shape:
            `(number of components, shape[1], shape[2], ...)`.
    """

    def __init__(self,
                 shape,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(AddComponents, self).__init__(**kwargs)
        self.shape = tuple(shape)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    def build(self, input_shape):
        self.components = self.add_weight(shape=self.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          name='components')
        self.built = True

    def call(self, inputs, **kwargs):
        output = inputs + self.components

        return output

    def compute_output_shape(self, input_shape):
        return self.shape
