# -*- coding: utf-8 -*-
"""Implementations of constraints.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.constraints import Constraint

import keras.backend as K


class EuclideanNormalization(Constraint):
    """Normalization to Euclidean norm of one.

    Normalizes a given input tensor to an Euclidean norm of one regarding
    the specified `axis` argument.

    # Arguments
        axis: An integer or tuple/list of integers, specifying the
            axis for the normalization

    # Input shape
        tensor with arbitrary shape

    # Output shape
        tensor with the same shape as the input tensor
    """

    def __init__(self,
                 axis=(0, 1, 2)):
        self.axis = axis

    def __call__(self, w):
        w /= K.sqrt(K.maximum(K.sum(K.square(w),
                                    self.axis,
                                    keepdims=True),
                              K.epsilon()))
        return w


class Clip(Constraint):
    """Clipping of all tensor values into the specified interval.

    Clipping of the values into the interval [min, max].

    # Arguments
        min_value: lower bound of the interval
        max_value: upper bound of the interval

    # Input shape
        tensor with arbitrary shape

    # Output shape
        tensor with the same shape as the input tensor
    """

    def __init__(self,
                 min_value=0.,
                 max_value=1.):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        w = K.clip(w, self.min_value, self.max_value)
        return w


def euclidean_normalization(w):
    return EuclideanNormalization()(w)


def clip(w):
    return Clip()(w)
