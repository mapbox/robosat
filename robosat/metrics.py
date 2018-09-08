"""Metrics for segmentation.
"""

import torch
import numpy as np


class MeanIoU:
    """Tracking mean intersection over union.
    """

    def __init__(self, labels):
        """Creates an new `MeanIoU` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels
        self.confusion_matrix = None

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """

        confusion = predicted / actual

        tn = torch.sum(torch.isnan(confusion)).item()
        fn = torch.sum(confusion == float("inf")).item()
        fp = torch.sum(confusion == 0).item()
        tp = torch.sum(confusion == 1).item()

        matrix = np.array([[tn, fn], [fp, tp]])

        if self.confusion_matrix is None:
            self.confusion_matrix = matrix
        else:
            self.confusion_matrix += matrix

    def get(self):
        """Retrieves the mean intersection over union score.

        Returns:
          The mean intersection over union score for all obersations seen so far.
        """

        intersection = np.diag(self.confusion_matrix)

        actual = self.confusion_matrix.sum(axis=1)
        predicted = self.confusion_matrix.sum(axis=0)

        union = actual + predicted - intersection

        iou = intersection / union.astype(np.float32)
        return np.nanmean(iou)


# Todo:
# - implement pixel accuracy
# - implement iou on a per-class basis (not mean'ed)

# intersection = np.diag(self.confusion_matrix)
# accuracy = intersection.sum() / self.confusion_matrix.sum()
# mean_cls_accuracy = np.nanmean(intersection / self.confusion_matrix.sum(axis=1))
