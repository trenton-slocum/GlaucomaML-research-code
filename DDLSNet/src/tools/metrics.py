"""Functions that help with model evaluation."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, roc_auc_score)


def get_predictions_and_labels(model, dataset):
    """Generate pairs of predictions and true labels given model and dataset."""
    labels = tf.math.argmax(np.concatenate([y for _, y in dataset], axis=0),
                            axis=1)
    vector_predictions = model.predict(dataset)

    return labels, vector_predictions


def gen_confusion_matrix(model, dataset, num_classes):
    """Generates a confusion matrix for the given model and dataset.
       Intended for multi-class or binary classification.
    """

    labels, vector_predictions = get_predictions_and_labels(model, dataset)

    int_predictions = tf.math.argmax(vector_predictions, axis=1)

    result = tf.math.confusion_matrix(labels,
                                      int_predictions,
                                      num_classes=num_classes)
    return result


def get_accuracy_scores(model, dataset):
    """Generates accuracy scores for the given model and dataset.
       Intended for non multi-label situations.
    """

    labels, vector_predictions = get_predictions_and_labels(model, dataset)

    int_predictions = tf.math.argmax(vector_predictions, axis=1)

    results = {}

    results['balanced'] = balanced_accuracy_score(labels, int_predictions)
    results['raw'] = accuracy_score(labels, int_predictions)

    return results


def get_auc_scores(model, dataset):
    """Generates multiclass ROC_AUC scores for the given model and dataset.
       Intended for multiclass situations.
    """
    y_true, y_pred = get_predictions_and_labels(model, dataset)

    average_types = ['macro', 'weighted']

    results = {
        average: roc_auc_score(y_true,
                               y_pred,
                               multi_class='ovr',
                               average=average)
        for average in average_types
    }

    return results


def get_classification_report(model, dataset):
    """Generates classification report for the given model and dataset.
       Intended for multiclass situations.
    """
    y_true, vector_predictions = get_predictions_and_labels(model, dataset)

    y_pred = tf.math.argmax(vector_predictions, axis=1)

    target_names = ['small', 'medium', 'large']

    return classification_report(y_true, y_pred, target_names=target_names)
