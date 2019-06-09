# -*- coding: utf-8 -*-
"""Implementations of losses.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K


class MarginLoss(object):
    """Margin loss.

    This loss is the margin loss implementation with a specifiable `margin`
    value.

    # Arguments
        margin: float, specifying the margin (probability gap).
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    """

    def __init__(self, margin=0.3):
        self.margin = K.variable(margin, name='margin')

    def __call__(self, y_true, y_pred):
        dp = K.sum(y_true * y_pred, axis=-1)
        dm = K.max(y_pred - y_true, axis=-1)
        return K.relu(dm - dp + self.margin)


def margin_loss(y_true, y_pred):
    return MarginLoss()(y_true, y_pred)


def elu_loss(y_true, y_pred):
    """ELU loss.

    This loss is the probability gap activated by the ELU activation.

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    """
    dp = K.sum(y_true * y_pred, axis=-1)
    dm = K.max(y_pred - y_true, axis=-1)
    return K.elu(dm - dp)
