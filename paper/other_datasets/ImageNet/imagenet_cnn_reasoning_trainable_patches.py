# -*- coding: utf-8 -*-
"""Implementation of the CBC for ImageNet with a fixed ResNet with 50 layers
as feature extractor for the CBC.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import argparse
import os

import tensorflow as tf

from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import Model
from keras import callbacks
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input

from keras_cbc.layers.component_input import ConstantInput, AddComponents
from keras_cbc.layers.detection_probability_functions import CosineSimilarity2D
from keras_cbc.layers.reasoning_layers import Reasoning
from keras_cbc.utils.losses import MarginLoss
from paper.other_datasets.ImageNet.utils.multi_gpu_siamese import \
    multi_gpu_siamese


def get_data_generators(args):
    """The ImageNet dataset is to large to completely load into memory. To load
    the training and test images we therefore use a image data generator.
    This function initializes the generators and returns them to be used
    during training.

    # Arguments
        args: Namespace with the arguments specifying the paths to the
        folders containing train and test images.

    # Returns
        Two data generators, one for training data and one for test data
    """
    train_datagen = ImageDataGenerator(dtype='float32',
                                       preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(
        args.train_path,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='custom_imagenet')

    val_datagen = ImageDataGenerator(dtype='float32',
                                     preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(
        args.test_path,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='custom_imagenet')

    return train_generator, val_generator


class Backbone(object):
    """Feature extractor of a CBC.

    For the experiments on ImageNet a non-trainable ResNet50 model is used as
    feature extractor. The ResNet used is the default implementation provided
    by Keras, with the top removed.

    The pretrained weights for the ResNet are also provided by Keras. The
    weights are downloaded the first time the network is used.
    """

    def __init__(self, img_input):
        resnet = keras.applications.ResNet50(input_tensor=img_input,
                                             include_top=False)
        resnet.summary()

        for layer in resnet.layers:
            layer.trainable = False

        self.model = resnet

    def __call__(self, *args, **kwargs):
        return self.model.output


def get_init_components_and_reasoning(args,
                                      component_shape,
                                      n_components,
                                      n_classes):
    """The patch components are initialized by cropping the center of 5 images
    from each class. Additionally, the initial reasoning matrix is biased
    towards the respective class of each component.

    # Arguments
        args: Namespace, with the argument for the test and train path
        component_shape: List/tuple of three integers, specifying the shape of
               the components (rows, cols, channels).
        n_components: Integer number of components
        n_classes: Integer number of classes

    # Returns:
        The component and reasoning initializer as numpy array.
    """
    reasoning = np.zeros((2, 1, n_components * n_classes, n_classes))
    reasoning[0] = np.random.random(
        (1, 1, n_components * n_classes, n_classes)) / 4
    reasoning[1] = 1 - reasoning[0]

    # Locate all classes
    classes = []
    for subdir in sorted(os.listdir(args.train_path)):
        if os.path.isdir(os.path.join(args.train_path, subdir)):
            classes.append(subdir)

    # Define the cropping size for each component
    components = np.zeros(
        (n_components * n_classes, component_shape[0], component_shape[1],
         component_shape[2]),
        dtype='uint8')
    left = (224 - component_shape[0]) / 2
    top = (224 - component_shape[1]) / 2
    right = (224 + component_shape[0]) / 2
    bottom = (224 + component_shape[1]) / 2

    # Load images
    image_idx = 0
    for class_idx, class_label in enumerate(classes):
        folder_path = os.path.join(args.train_path, class_label)

        # iterate over images
        image_names = os.listdir(folder_path)
        for i in range(0, n_components):
            file_path = os.path.join(folder_path, image_names[i])

            if os.path.isfile(file_path):
                proto = keras.preprocessing.image.load_img(
                    file_path,
                    target_size=(224, 224),
                    interpolation='custom_imagenet').crop(
                    (left, top, right, bottom))
                components[image_idx, :, :, :] = proto

                # Initialize reasoning
                tmp = reasoning[0, 0, image_idx, class_idx]
                reasoning[0, 0, image_idx, class_idx] = reasoning[
                    -1, 0, image_idx, class_idx]
                reasoning[-1, 0, image_idx, class_idx] = tmp
                image_idx = image_idx + 1

    components = preprocess_input(components)

    return components, reasoning


def model(input_shape,
          component_shape,
          n_classes,
          components_initializer=None,
          reasoning_initializer=None):
    """Defines the CBC model without trainable backbone. The model is
    instantiated on the CPU and then parallelized over the specified GPUs.

       # Arguments:
           input_shape: List/tuple of three integers, specifying the shape of
               the input data (rows, cols, channels).
           n_classes: Integer, specifying the number of classes.
           component_shape: List/tuple of three integers, specifying the shape
               of a component (rows, cols, channels).
           components_initializer: Numpy array, containing the initial
                components in the input space
           reasoning_initializer: Numpy array, the initialized reasoning matrix

       # Returns:
           The parallelized model
       """
    with tf.device('/cpu:0'):
        # Initialize backbone
        data_input = Input(shape=input_shape, name='model_input')
        backbone = Backbone(data_input)
        data_output = backbone()

        # Create initial components by processing them through the feature
        # extractor.
        patch_input = Input(shape=component_shape, name='model_input')
        patch_network = Backbone(patch_input)
        init_components = patch_network.model.predict(components_initializer)

        # Create component input
        components_input = ConstantInput(np.zeros((1,)), name='components')()
        components = AddComponents(shape=init_components.shape,
                                   initializer=lambda x: init_components,
                                   )(components_input)

        # Create detection network
        detection = CosineSimilarity2D(padding='valid', activation='relu')(
            [data_output, components])

        # Down sample to get detection probability for each component
        detection = GlobalMaxPool2D()(detection)

        # Reasoning
        reasoning = Reasoning(
            n_classes=n_classes,
            reasoning_initializer=lambda x: reasoning_initializer)

        probabilities = reasoning(detection)

        # Create model
        single_model = Model([data_input, components_input], probabilities)
        single_model.summary()

    gpus_model = multi_gpu_siamese(single_model, gpus=len(
        os.environ["CUDA_VISIBLE_DEVICES"].split(',')))

    return gpus_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        help="Load h5 model trained weights")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate of Adam.")
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--gpu', default='0,1', type=str,
                        help="Select the GPU used for training.")
    parser.add_argument('--eval', action='store_true',
                        help="Evaluation mode: statistics of the trained "
                             "model.")
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to the ImageNet training data folder '
                             'where the sub-folders are the folders '
                             'containing all the images from one class.')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to the ImageNet test data folder '
                             'where the sub-folders are the folders '
                             'containing all the images from one class.')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n_components_per_class = 5
    n_classes = 1000
    input_shape = (224, 224, 3)
    component_shape = (64, 64, 3)

    init_components, init_reasoning = get_init_components_and_reasoning(
        args,
        component_shape,
        n_components_per_class,
        n_classes)

    train_model = model(input_shape,
                        component_shape,
                        n_classes,
                        components_initializer=init_components,
                        reasoning_initializer=init_reasoning)

    train_model.summary()

    if args.weights:
        train_model.load_weights(args.weights)

    train_model.compile(optimizer=Adam(lr=args.lr),
                        loss=MarginLoss(0.1),
                        metrics=[categorical_accuracy,
                                 top_k_categorical_accuracy])

    train_generator, val_generator = get_data_generators(args)
    n_images_train = 1281166
    n_images_val = 50000

    if not args.eval:
        # Callbacks
        checkpoint = callbacks.ModelCheckpoint(
            args.save_dir + '/weights-{epoch:02d}.h5',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
        lr_reduce = callbacks.ReduceLROnPlateau(factor=0.9,
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                patience=5)

        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=n_images_train / args.batch_size,
            epochs=args.epochs,
            validation_data=val_generator,
            validation_steps=n_images_val / args.batch_size,
            callbacks=[checkpoint,
                       lr_reduce,
                       csv_logger],
            verbose=1)

    print('train results:')
    print(train_model.evaluate_generator(
        train_generator,
        steps=n_images_train / args.batch_size,
        verbose=True))
    print('test results:')
    print(train_model.evaluate_generator(
        val_generator,
        steps=n_images_val / args.batch_size,
        verbose=True))

    train_model.save_weights(args.save_dir + '/trained_model.h5')
