# -*- coding: utf-8 -*-
"""Implementation of the alpha and diameter CBC from the paper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os

from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.initializers import RandomUniform

from keras_cbc.layers.component_input import \
    ConstantInput, AddComponents
from keras_cbc.layers.detection_probability_functions import \
    CosineSimilarity2D
from keras_cbc.layers.reasoning_layers import Reasoning2D
from keras_cbc.utils.constraints import \
    euclidean_normalization, clip
from keras_cbc.utils.losses import margin_loss
from keras_cbc.utils.activations import swish
from keras_cbc.visualizations.basic_visualizations import \
    plot_components, plot_components_heatmaps
from keras_cbc.visualizations.input_dependent import \
    plot_input_reasoning_heatmaps, plot_input_reconstruction
from keras_cbc.visualizations.input_independent import \
    plot_optimal_reconstruction, plot_optimal_reasoning_heatmaps
from keras_cbc.visualizations.utils import make_uint8_img
from keras_cbc.utils.evaluation import statistics

import cv2


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


class Backbone(object):
    """Feature extractor of a CBC with support of the special Siamese
    training procedure.

    We use this class to realize our special training setting where the
    gradient back flow to feature extractor weights is just realized over the
    Siamese network path where input images are preprocessed. For the
    component processing Siamese path no gradient back flow to feature
    extractor layer weights is realized and the gradient is only pushed to
    the components. Via this implementation we can train the components in
    the input space.

    If a 'dependent_backbone' is given, then it is assumed that they share
    the same architecture. In this case the layers are initialized as non-
    trainable and as dependent on the weights of the given backbone. The
    dependency is realized by copying all the weights at the beginning of
    each batch from the 'dependent_backbone' to the own weights. The method
    'copy_weights' is used for that in combination with a custom callback.

    Keep care if you use batch normalization as it requires a special
    treatment.

    We know that the copying of weights is not efficient in terms of
    computational overhead. But we see now other workaround to realize that
    in Keras.
    """

    def __init__(self, dependent_backbone=None):
        self.dependent_backbone = dependent_backbone
        if dependent_backbone is not None:
            self.trainable = False
        else:
            self.trainable = True

        self.layers = [Conv2D(filters=32,
                              kernel_size=(5, 5),
                              activation=swish,
                              kernel_constraint=euclidean_normalization,
                              use_bias=True,
                              padding='valid',
                              trainable=self.trainable),
                       Conv2D(filters=64,
                              kernel_size=(3, 3),
                              activation=swish,
                              kernel_constraint=euclidean_normalization,
                              use_bias=True,
                              padding='valid',
                              trainable=self.trainable)]

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def copy_weights(self):
        for i, layer in enumerate(self.dependent_backbone.layers):
            self.layers[i].set_weights(layer.get_weights())


def model(input_shape,
          n_classes,
          n_components,
          component_shape,
          use_pixel_probabilities):
    """Defines the CBC model in the Siamese setting.

    # Arguments:
        input_shape: List/tuple of three integers, specifying the shape of
            the input data (rows, cols, channels).
        n_classes: Integer, specifying the number of classes.
        n_components: Integer, specifying the number of trainable components
            in the input space.
        component_shape: List/tuple of three integers, specifying the shape
            of a component (rows, cols, channels).
        use_pixel_probabilities: Boolean, specifying to train with trainable
            pixel probabilities or not.

    # Returns:
        A list of two objects:
            [0]: keras CBC model
            [1] backbone with non trainable weights which must be passed
                to the callback.
    """
    # initialize the two paths of the Siamese network
    backbone = Backbone()
    backbone_fix = Backbone(dependent_backbone=backbone)

    # initialize the two input sources for the Siamese network
    data_input = Input(shape=input_shape, name='model_input')
    components_input = ConstantInput(np.zeros((1,)), name='components')()

    # call input data processing path of the Siamese network
    data_output = backbone(data_input)

    # call component processing path of the Siamese network
    add_components = AddComponents(shape=(n_components,) + component_shape,
                                   initializer=RandomUniform(0.45, 0.55),
                                   constraint=clip,
                                   name='add_components_1')
    components = add_components(components_input)
    components_output = backbone_fix(components)

    # CBC layers: detection followed by reasoning
    detection = CosineSimilarity2D(
        padding='valid', activation='relu')([data_output, components_output])

    detection = MaxPool2D(pool_size=(3, 3),
                          strides=3,
                          padding='valid')(detection)

    reasoning = Reasoning2D(
        n_classes=n_classes,
        n_replicas=2,
        reasoning_initializer=RandomUniform(0, 1),
        use_pixel_probabilities=use_pixel_probabilities,
        name='reasoning2d_1')

    probabilities = reasoning(detection)

    # Winner-Take-All over replicas
    probabilities = Lambda(lambda x: K.max(x, -1))(probabilities)

    # Squeeze one dimensions
    probabilities = Lambda(
        lambda x: K.squeeze(K.squeeze(x, 1), 1))(probabilities)

    # we have to return the fixed_backbone to create the callback
    return Model([data_input, components_input], probabilities), backbone_fix


def eval_plots(eval_model, x_test, resize_factor, save_dir):
    """Evaluation plots of the eval_model on the given test set.

    This function plots all the presented visualizations of a spatial
    reasoning model presented in the paper. Maximal 10 samples of x_test are
    evaluated.

    # Arguments:
        eval_model: Keras CBC model which should be evaluated
        x_test: dataset which should be evaluated.
        resize_factor: Integer, specifying the resizing of the spatial
            reasoning kernel before the reconstruction. This is usually
            equivalent to the pool size after the detection probability
            function.
        save_dir: String, specifying the output path.
    """

    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    n_test = 10
    x_test = x_test[:n_test]
    save_dir = save_dir + '/eval/'

    # --- get components ---
    components = eval_model.get_layer('add_components_1').get_weights()[0]

    # --- preprocess reasoning ---
    reasoning_layer = eval_model.get_layer('reasoning2d_1')
    plot_model = Model(eval_model.input,
                       [reasoning_layer.input, reasoning_layer.output])
    detection_probability, class_probabilities = plot_model.predict(x_test)

    # get weights of the reasoning layer
    weights = reasoning_layer.get_weights()
    if len(weights) == 3:
        reasoning_weights, pixel_probabilities, component_probabilities = \
            weights
    elif len(weights) == 2:
        if reasoning_layer.use_pixel_probabilities:
            reasoning_weights, pixel_probabilities = weights
            component_probabilities = None
        else:
            reasoning_weights, component_probabilities = weights
            pixel_probabilities = None
    else:
        reasoning_weights = weights[0]
        pixel_probabilities = None
        component_probabilities = None

    # decode reasoning_weights
    positive_reasoning = reasoning_weights[0]
    negative_reasoning = (1 - positive_reasoning) * reasoning_weights[1]

    n_components = negative_reasoning.shape[2]
    n_classes = reasoning_layer.n_classes
    n_replicas = reasoning_layer.n_replicas

    input_shape = x_test.shape[1:]

    # shape of the spatial reasoning kernel
    kernel_shape = negative_reasoning.shape[:2]
    # resized kernel shape before reconstruction
    resized_kernel_shape = \
        (resize_factor * kernel_shape[0], resize_factor * kernel_shape[1])

    # receptive field shape of the spatial reasoning process in the input space
    resized_reasoning_shape = \
        (input_shape[0] - class_probabilities.shape[1] + 1,
         input_shape[1] - class_probabilities.shape[2] + 1)

    # reshape to class x replicas
    positive_reasoning = np.reshape(positive_reasoning,
                                    kernel_shape +
                                    (n_components, n_classes, n_replicas))
    negative_reasoning = np.reshape(negative_reasoning,
                                    kernel_shape +
                                    (n_components, n_classes, n_replicas))

    # expand shape if no multiple reasoning was used to have a constant shape
    if n_replicas == 1:
        class_probabilities = np.expand_dims(class_probabilities, -1)

    # extract winner reasoning index
    tmp = np.transpose(class_probabilities, (0, 1, 2, 4, 3))
    tmp = np.reshape(tmp, (n_test, -1, n_classes))
    wta_idx = np.argmax(tmp, axis=1)
    wta_idx = np.unravel_index(wta_idx,
                               class_probabilities.shape[1:3] + (n_replicas,))

    # decode pixel probabilities
    if pixel_probabilities is None:
        pixel_probabilities = np.ones(
            kernel_shape + (1, n_classes, n_replicas))
    else:
        pixel_probabilities = np.reshape(
            pixel_probabilities, kernel_shape + (1, n_classes, n_replicas))

    # create pixel heatmaps
    pixel_probabilities = softmax(pixel_probabilities, axis=(0, 1))
    pixel_heatmaps = pixel_probabilities / \
                     np.max(pixel_probabilities, axis=(0, 1), keepdims=True)

    # effective reasoning probabilities
    if component_probabilities is not None:
        component_probabilities = softmax(component_probabilities, axis=2)
        component_probabilities = np.expand_dims(component_probabilities, -1)
        positive_reasoning = positive_reasoning * component_probabilities
        negative_reasoning = negative_reasoning * component_probabilities

    reasoning_norm = np.sum(positive_reasoning + negative_reasoning,
                            axis=2,
                            keepdims=True)
    positive_effective_reasoning = positive_reasoning / reasoning_norm
    negative_effective_reasoning = negative_reasoning / reasoning_norm

    # --- plots ---
    # test_samples
    path = save_dir + '/samples/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, img in enumerate(x_test):
        cv2.imwrite(path + str(i) + '.png', make_uint8_img(img))

    plot_components(components, save_dir + '/components/')

    plot_components_heatmaps(detection_probability, x_test,
                             save_dir + '/comp_heatmaps/')

    plot_input_reasoning_heatmaps(positive_effective_reasoning,
                                  negative_effective_reasoning,
                                  detection_probability,
                                  pixel_heatmaps,
                                  wta_idx,
                                  resized_reasoning_shape,
                                  x_test,
                                  save_dir + '/reasoning_input_heatmaps/')

    plot_input_reconstruction(components,
                              positive_effective_reasoning,
                              negative_effective_reasoning,
                              detection_probability,
                              pixel_heatmaps,
                              wta_idx,
                              resized_kernel_shape,
                              x_test,
                              save_dir + '/reasoning_input_reconstruction/')

    plot_optimal_reasoning_heatmaps(positive_effective_reasoning,
                                    negative_effective_reasoning,
                                    pixel_heatmaps,
                                    resized_reasoning_shape,
                                    save_dir + '/reasoning_optimal_heatmap/')

    plot_optimal_reconstruction(components,
                                positive_effective_reasoning,
                                negative_effective_reasoning,
                                pixel_heatmaps,
                                resized_kernel_shape,
                                save_dir +
                                '/reasoning_optimal_reconstruction/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        help="Load h5 model trained weights.")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=0.003, type=float,
                        help="Initial learning rate of Adam.")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu', default=0, type=int,
                        help="Select the GPU used for training.")
    parser.add_argument('--eval', action='store_true',
                        help="Evaluation mode: visualizing the trained model.")
    parser.add_argument('--eval_data', default='', type=str,
                        help="Provide a path to images which should be "
                             "evaluated. Otherwise the test dataset is used. "
                             "The method expects a *.npy file and assumes "
                             "that the images are preprocessed equivalent to "
                             "the train/test data")
    parser.add_argument('--use_pixel_probabilities', action='store_true',
                        help="Train with pixel probabilities. Set this "
                             "argument for the alpha-CBC. Otherwise it is "
                             "equivalent to the diameter-CBC")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = get_data()

    train_model, backbone_fix = model(
        input_shape=x_test.shape[1:],
        n_classes=10,
        n_components=8,
        component_shape=(7, 7, 1),
        use_pixel_probabilities=args.use_pixel_probabilities)

    if args.weights:
        train_model.load_weights(args.weights)

    train_datagen = ImageDataGenerator(width_shift_range=2,
                                       height_shift_range=2,
                                       rotation_range=15)

    generator = train_datagen.flow(x_train,
                                   y_train,
                                   batch_size=args.batch_size)

    train_model.summary()

    train_model.compile(optimizer=Adam(lr=args.lr),
                        loss=margin_loss,
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
        copy_weights = callbacks.LambdaCallback(
            on_batch_begin=lambda batch, logs: backbone_fix.copy_weights())

        callbacks = [checkpoint, lr_reduce, csv_logger, copy_weights]

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

    if args.eval_data == '':
        x_eval = x_test
    else:
        x_eval = np.load(args.eval_data)

    # create interpretation plots on the eval dataset
    # resize_factor is chosen equivalent to the pooling size after the
    # detection probability function
    eval_plots(train_model, x_eval, resize_factor=3, save_dir=args.save_dir)
