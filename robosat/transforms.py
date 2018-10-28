"""PyTorch-compatible transformations.
"""

import random
import torch
import cv2
import numpy as np


class ImageToTensor:
    """Callable to convert a OpenCV H,W,C image into a PyTorch tensor.
    """

    def __call__(self, image):
        """Converts the image into a tensor.

        Args:
          image: the image to convert into a PyTorch tensor.

        Returns:
          The converted PyTorch tensor.
        """

        return torch.from_numpy(np.moveaxis(image, 2, 0)).float()


class MaskToTensor:
    """Callable to convert an OpenCV H,W image into a PyTorch tensor.
    """

    def __call__(self, mask):
        """Converts the mask into a tensor.

        Args:
          mask: the mask to convert into a PyTorch tensor.

        Returns:
          The converted PyTorch tensor.
        """

        return torch.from_numpy(mask).long()


class JointCompose:
    """Callable to transform an image and it's mask at the same time.
    """

    def __init__(self, transforms):
        """Creates an `JointCompose` instance.

        Args:
          transforms: list of tuple with (image, mask) transformations.
        """

        self.transforms = transforms

    def __call__(self, image, mask):
        """Applies multiple transformations to the image and its mask at the same time.

        Args:
          image: the image to transform.
          mask: the mask to transform.

        Returns:
          The transformed (image, mask) tuple.
        """

        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class JointTransform:
    """Callable to compose non-joint transformations into joint-transformations on images and mask.

    Note: must not be used with stateful transformations (e.g. rngs) which need to be in sync for image and mask.
    """

    def __init__(self, image_transform, mask_transform):
        """Creates an `JointTransform` instance.

        Args:
          image_transform: the transformation to run on the image or `None` for no-op.
          mask_transform: the transformation to run on the mask or `None` for no-op.

        Returns:
          The (image, mask) tuple with the transformations applied.
        """

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        """Applies the transformations associated with image and its mask.

        Args:
          image: the image to transform.
          mask: the mask to transform.

        Returns:
          The (image, mask) tuple with the transformations applied.
        """

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class JointRandomFlipOrRotate:
    """Callable to randomly rotate image and its mask.
    """

    def __init__(self, p):
        """Creates an `JointRandomRotation` instance.

        Args:
          p: the probability for rotating.
        """
        assert p >= 0.0 and p <= 1.0, "Probability must be expressed in 0-1 interval"
        self.p = p

    def __call__(self, image, mask):
        """Randomly rotates or flip image and its mask.

        Args:
          image: the image to transform.
          mask: the mask to transform.

        Returns:
          The (image, mask) tuple with either image and mask flip or rotated or kept untouched (but synced)
        """

        if random.random() > self.p:
            return image, mask

        transform = random.choice(["Rotate90", "Rotate180", "Rotate270", "HorizontalFlip", "VerticalFlip"])

        if transform == "Rotate90":
            return cv2.flip(cv2.transpose(image), +1), cv2.flip(cv2.transpose(mask), +1)
        elif transform == "Rotate180":
            return cv2.flip(image, -1), cv2.flip(mask, -1)
        elif transform == "Rotate270":
            return cv2.flip(cv2.transpose(image), 0), cv2.flip(cv2.transpose(mask), 0)
        elif transform == "HorizontalFlip":
            return cv2.flip(image, +1), cv2.flip(mask, +1)
        elif transform == "VerticalFlip":
            return cv2.flip(image, 0), cv2.flip(mask, 0)


class JointResize:
    """Callable to resize image and its mask
    """

    def __init__(self, size):
        """Creates an `JointResize` instance.

        Args:
          size: the desired square side size
        """
        self.hw = (size, size)

    def __call__(self, image, mask):
        """Resize image and its mask

        Args:
          image: the image to transform.
          mask: the mask to transform.

        Returns:
          The (image, mask) tuple resized
        """

        if self.hw == image.shape[0:2]:
            pass
        elif self.hw[0] < image.shape[0] and self.hw[1] < image.shape[1]:
            image = cv2.resize(image, self.hw, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, self.hw, interpolation=cv2.INTER_LINEAR)

        if self.hw != mask.shape:
            mask = cv2.resize(mask, self.hw, interpolation=cv2.INTER_NEAREST)

        return image, mask
