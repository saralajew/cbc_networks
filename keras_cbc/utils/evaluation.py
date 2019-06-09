# -*- coding: utf-8 -*-
"""Implementation of evaluation methods.
"""
import numpy as np


def statistics(x_train, y_train, x_test, y_test, eval_model, path):
    """Function to compute the average prediction probability and
    probability gap over correct and incorrect classifications.

    The statistics is computed over the test dataset.

    # Arguments:
        x_train: Numpy array, training data of the model.
        y_train: Numpy array, training labels of the model.
        x_test: Numpy array, test data of the model.
        y_test: Numpy array, test labels of the model.
        eval_model: Keras model which should be evaluated.
        path: String, output path.
    """

    # preprocessing
    y_pred = eval_model.predict(x_test)

    y_true = np.max(y_pred * y_test, -1)
    y_best_matching_false = np.max(y_pred - y_test, -1)
    y_signed_gap = y_true - y_best_matching_false

    incorrect_classifications = y_signed_gap < 0
    correct_classifications = y_signed_gap >= 0

    # create statistics
    with open(path, 'w') as f:
        def print_and_write(text, value, std=None):
            if std is None:
                f.write('\n' + text + str(value))
                print(text + str(value))
            else:
                f.write('\n' + text + str(value) + ' +/- ' + str(std))
                print(text + str(value) + ' +/- ' + str(std))

        print_and_write('train results (loss, acc): ',
                        eval_model.evaluate(x_train, y_train))
        print_and_write('test results (loss, acc): ',
                        eval_model.evaluate(x_test, y_test))

        # we use this as sanity check that our evaluation of incorrect
        # predictions by the probability gap is right. This value must be
        # equivalent to (1 - test_acc)!
        print_and_write('error rate by probability gap (sanity check): ',
                        np.sum(incorrect_classifications) / x_test.shape[0])

        print_and_write('\n0--- statistics correct classifications ---', 0)
        print_and_write('average predicted classification probability: ',
                        np.mean(y_true[correct_classifications]),
                        np.std(y_true[correct_classifications]))
        print_and_write('average best-matching incorrect probability: ',
                        np.mean(
                            y_best_matching_false[correct_classifications]),
                        np.std(y_best_matching_false[correct_classifications]))
        print_and_write('average probability gap between them: ',
                        np.mean(y_signed_gap[correct_classifications]),
                        np.std(y_signed_gap[correct_classifications]))

        print_and_write('\n0--- statistics incorrect classifications ---', 0)
        print_and_write('average predicted classification probability: ',
                        np.mean(
                            y_best_matching_false[incorrect_classifications]),
                        np.std(
                            y_best_matching_false[incorrect_classifications]))
        print_and_write('average probability of the true class:',
                        np.mean(y_true[incorrect_classifications]),
                        np.std(y_true[incorrect_classifications]))
        print_and_write('average probability gap between them: ',
                        np.mean(-y_signed_gap[incorrect_classifications]),
                        np.std(y_signed_gap[incorrect_classifications]))
