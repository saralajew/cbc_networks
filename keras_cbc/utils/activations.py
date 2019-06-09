# -*- coding: utf-8 -*-
"""Implementations of activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K


def swish(x):
    """Swish activation function.

    Applies the swish activation on the input tensor.

    # Arguments
        x: Input tensor

    # Input shape
        tensor with arbitrary shape

    # Output shape
        tensor with the same shape as the input tensor
    """
    return x*K.sigmoid(x)


def squared_relu(x):
    """Squared ReLU activation function.

    Applies the squared ReLU activation on the input tensor.

    # Arguments
        x: Input tensor

    # Input shape
        tensor with arbitrary shape

    # Output shape
        tensor with the same shape as the input tensor
    """
    return K.square(K.relu(x))
