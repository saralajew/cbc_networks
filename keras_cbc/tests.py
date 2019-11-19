import pytest

import numpy as np
import tensorflow as tf

from keras.layers import Input
from keras.models import Model

from keras_cbc.layers.detection_probability_functions import CosineSimilarity2D


def test_cosine_similarity_2d():

    def cos(a, b):

        batch = tf.convert_to_tensor(a, dtype=tf.float32)
        components = tf.convert_to_tensor(b, dtype=tf.float32)

        inputs_b = Input(batch_shape=batch.shape)
        inputs_c = Input(batch_shape=components.shape)

        detection = CosineSimilarity2D(
            padding='valid', activation='relu')([inputs_b, inputs_c])

        model = Model(inputs=[inputs_b, inputs_c], outputs=[detection])

        out = model.predict([batch, components], steps=1)

        assert out.shape == (1, 1, 1, 1)

        # batch/components, W, H, channels
    cos(np.ones((1, 24, 24, 3)),            np.ones((1, 24, 24, 3)))
    cos(np.zeros((1, 24, 24, 3)),           np.zeros((1, 24, 24, 3)))
    cos(np.random.random((1, 24, 24, 1)),   np.random.random((1, 24, 24, 1)))
