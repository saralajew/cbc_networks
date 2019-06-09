# -*- coding: utf-8 -*-
"""Utility functions to load the GTSRB dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import cv2
import os

import numpy as np


class LabelsOfTestImage:
    """Given a file name of a test sample this function returns the class
    label of the class that it belongs to.

    # Arguments:
        image_name: String, name to the test sample of which the class label
            is requested in the format <name>.ppm
        cvs_path: String, path to the csv file containing the mapping from
            file names to class labels. That the official csv file
            containing the annotations.

    # Returns:
        Integer, the class label to the image_name.
    """
    def __init__(self,
                 csv_path):

        # build dictionary of file_names (string) --> labels (int)
        self.dictionary = {}
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)

            for i, row in enumerate(reader):
                # remove header
                if i == 0:
                    continue

                # store name label pair
                self.dictionary.update(
                    {row[0].split(';')[0]: int(row[0].split(';')[7])})

    def __call__(self, image_name):
        return self.dictionary[image_name]


def test_data_with_label(image_path, csv_path):
    """Loads all test images found at the supplied location in the local
    file system. All images are assumed to be present in one folder, with a
    csv file available to dictate to which class each file belongs.

    The method resizes all the images to 64x64 by opencv's bilinear
    interpolation procedure.

    # Arguments
        image_path: String, path to the folder containing the test images.
        cvs_path: String, path to the csv file containing the mapping from
            file name to class label

    # Returns:
        First return: Numpy array, the test images in the shape
            (n_test, 64, 64, 3) with 'dtype' uint8.
        Second return: Numpy array, the test labels in the shape
            (n_test,) with 'dtype' 'uint64'.
    """
    print("Starting loading of test data")

    test_images = []
    test_labels = []

    image_name_label_dictionary = LabelsOfTestImage(csv_path)

    for i in os.listdir(image_path):
        # Loop through all images
        path = os.path.join(image_path, i)

        if path.split('.')[1] == 'ppm':
            # get label
            image_name = path.split('/')[-1]
            label = image_name_label_dictionary(image_name)

            # read image and resize to 64 x 64
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

            test_images.append(np.array(img))
            test_labels.append(label)

    print("Finished loading test data")

    return np.asarray(test_images), np.asarray(test_labels).astype('uint64')


def train_data_with_label(train_path):
    """
    Loads all training images found at the supplied location in the local file
    system. The different classes are assumed to be separated into different
    folder.

    The method resizes all the images to 64x64 by opencv's bilinear
    interpolation procedure.

    # Arguments
        train_data: path to the folder containing the training images.

    # Returns:
        First return: Numpy array, the train images in the shape
            (n_train, 64, 64, 3) with 'dtype' uint8.
        Second return: Numpy array, the train labels in the shape
            (n_train,) with 'dtype' 'uint64'.
    """
    print("Starting loading of train data")

    train_images = []
    train_labels = []

    for k in os.listdir(train_path):
        # Loop through the different classes
        path_dir = os.path.join(train_path, k)

        for i in os.listdir(path_dir):
            # Loop through the different images
            path = os.path.join(path_dir, i)

            if path.split('.')[1] == 'ppm':
                # read image and resize to 64 x 64
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

                train_images.append(np.array(img))
                train_labels.append(int(k))

    print("Finished loading train data")

    return np.asarray(train_images), np.asarray(train_labels)
