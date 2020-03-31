# -*- coding: utf-8 -*-
"""Implementation of the baseline CNN on CIFAR-10 from the paper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from keras.layers import *
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.datasets import cifar10
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras_cbc.utils.evaluation import statistics


def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def model(input_shape,
          n_classes):
    """Defines the baseline CNN which is equivalent to the CBC in terms of
    the architecture.

     # Arguments:
        input_shape: List/tuple of three integers, specifying the shape of
            the input data (rows, cols, channels).
        n_classes: Integer, specifying the number of classes.

    # Returns:
        The Keras CNN model.
    """
    # initialize input
    data_input = Input(shape=input_shape, name='model_input')

    # this is the same backbone in the non-standard setting (with ReLU,
    # no constraint and batch normalization)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
               use_bias=True, padding='valid')(data_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               use_bias=True, padding='valid')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
               use_bias=True, padding='valid')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
               use_bias=True, padding='valid')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # these are the architectural equivalent layers to the CBC detection and
    # reasoning layer
    x = Conv2D(filters=10, kernel_size=(5, 5), activation='relu',
               use_bias=True, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    return Model(data_input, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        help="Load h5 model trained weights")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--epochs', default=475, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate of Adam.")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu', default=0, type=int,
                        help="Select the GPU used for training.")
    parser.add_argument('--eval', action='store_true',
                        help="Evaluation mode: statistics of the trained "
                             "model.")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = get_data()

    train_model = model(input_shape=x_test.shape[1:],
                        n_classes=10)

    train_model.summary()

    if args.weights:
        train_model.load_weights(args.weights)

    train_datagen = ImageDataGenerator(width_shift_range=2,
                                       height_shift_range=2,
                                       rotation_range=15)

    generator = train_datagen.flow(x_train,
                                   y_train,
                                   batch_size=args.batch_size)

    train_model.compile(optimizer=Adam(lr=args.lr),
                        loss=categorical_crossentropy,
                        metrics=['accuracy'])

    if not args.eval:
        # Callbacks
        checkpoint = callbacks.ModelCheckpoint(
            args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True,
            save_weights_only=True, verbose=1)
        csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
        lr_reduce = callbacks.ReduceLROnPlateau(factor=0.9,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                patience=5)

        callbacks = [checkpoint, lr_reduce, csv_logger]

        train_model.fit_generator(
            generator=generator,
            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
            epochs=args.epochs,
            validation_data=[x_test, y_test],
            callbacks=callbacks,
            verbose=1)

        train_model.save_weights(args.save_dir + '/trained_model.h5')

    # --- evaluation
    # compute statistics on the test dataset
    path = args.save_dir + '/statistics.txt'
    statistics(x_train, y_train, x_test, y_test, train_model, path)
