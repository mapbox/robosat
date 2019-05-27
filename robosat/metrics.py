"""Metrics for segmentation.
"""

import torch
import math
import numpy as np


class Metrics:
    """Tracking mean metrics
    """

    def __init__(self, labels):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """

        masks = torch.argmax(predicted, 0)
        confusion = masks.view(-1).float() / actual.view(-1).float()

        self.tn += torch.sum(torch.isnan(confusion)).item()
        self.fn += torch.sum(confusion == float("inf")).item()
        self.fp += torch.sum(confusion == 0).item()
        self.tp += torch.sum(confusion == 1).item()

    def get_miou(self):
        """Retrieves the mean Intersection over Union score.

        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        try:
            miou = np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp)])
        except ZeroDivisionError:
            miou = float("NaN")

        return miou

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.

        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """

        try:
            iou = self.tp / (self.tp + self.fn + self.fp)
        except ZeroDivisionError:
            iou = float("NaN")

        return iou

    def get_mcc(self):
        """Retrieves the Matthew's Coefficient Correlation score.

        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """

        try:
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
            )
        except ZeroDivisionError:
            mcc = float("NaN")

        return mcc


# Todo:
# - Rewrite mIoU to handle N classes (and not only binary SemSeg)
