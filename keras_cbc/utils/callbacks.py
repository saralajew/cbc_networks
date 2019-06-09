# -*- coding: utf-8 -*-
"""Implementation of callbacks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import Callback


class LossScheduler(Callback):
    """Callback for scheduling different loss functions.

        Recompiles the model at given epochs during the training phase with
        a different loss function.

        # Arguments
            losses: list of loss functions to be scheduled.
            epochs: list with the same length as losses containing the epoch
                at which each loss function should be scheduled.
            optimizer: optimizer that should be used for each re-compilation
                of the model.
            metrics: list of metrics to be used for each re-compilation of
                the model.
            reduce_lr_on_plateau: ReduceLROnPlateau Keras callback, which is
                reset after the loss is changed.
        """

    def __init__(self, losses, epochs, optimizer, metrics,
                 reduce_lr_on_plateau=None):
        super(Callback, self).__init__()

        assert(len(losses) == len(epochs))

        self.margins = losses
        self.epochs = epochs
        self.optimizer = optimizer
        self.metrics = metrics
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        print(self.epochs, self.margins)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.epochs:
            ind = self.epochs.index(epoch)
            new_loss = self.margins[ind]

            self.model.compile(optimizer=self.optimizer,
                               loss=new_loss,
                               metrics=self.metrics)
            if self.reduce_lr_on_plateau is not None:
                self.reduce_lr_on_plateau._reset()

            print("Loss function updated")
