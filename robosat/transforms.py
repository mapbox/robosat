"""PyTorch-compatible transformations.
"""

import random

import torch
import numpy as np
from PIL import Image

import torchvision


# Callable to convert a RGB image into a PyTorch tensor.
ImageToTensor = torchvision.transforms.ToTensor


class MaskToTensor:
    """Callable to convert a PIL image into a PyTorch tensor.
    """

    def __call__(self, image):
        """Converts the image into a tensor.

        Args:
          image: the PIL image to convert into a PyTorch tensor.

        Returns:
          The converted PyTorch tensor.
        """

        return torch.from_numpy(np.array(image, dtype=np.uint8)).long()


class ConvertImageMode:
    """Callable to convert a PIL image into a specific image mode (e.g. RGB, P)
    """

    def __init__(self, mode):
        """Creates an `ConvertImageMode` instance.

        Args:
          mode: the PIL image mode string
        """

        self.mode = mode

    def __call__(self, image):
        """Applies to mode conversion to an image.

        Args:
          image: the PIL.Image image to transform.
        """

        return image.convert(self.mode)


class JointCompose:
    """Callable to transform an image and it's mask at the same time.
    """

    def __init__(self, transforms):
        """Creates an `JointCompose` instance.

        Args:
          transforms: list of tuple with (image, mask) transformations.
        """

        self.transforms = transforms

    def __call__(self, images, mask):
        """Applies multiple transformations to the images and the mask at the same time.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The transformed PIL.Image (images, mask) tuple.
        """

        for transform in self.transforms:
            images, mask = transform(images, mask)

        return images, mask


class JointTransform:
    """Callable to compose non-joint transformations into joint-transformations on images and mask.

    Note: must not be used with stateful transformations (e.g. rngs) which need to be in sync for image and mask.
    """

    def __init__(self, image_transform, mask_transform):
        """Creates an `JointTransform` instance.

        Args:
          image_transform: the transformation to run on the images or `None` for no-op.
          mask_transform: the transformation to run on the mask or `None` for no-op.

        Returns:
          The (images, mask) tuple with the transformations applied.
        """

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, images, mask):
        """Applies the transformations associated with images and their mask.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with images and mask transformed.
        """

        if self.image_transform is not None:
            images = [self.image_transform(v) for v in images]

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return images, mask


class JointRandomVerticalFlip:
    """Callable to randomly flip images and its mask top to bottom.
    """

    def __init__(self, p):
        """Creates an `JointRandomVerticalFlip` instance.

        Args:
          p: the probability for flipping.
        """

        self.p = p

    def __call__(self, images, mask):
        """Randomly flips images and their mask top to bottom.

        Args:
          images: the PIL.Image image to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask flipped or none of them flipped.
        """

        if random.random() < self.p:
            return [v.transpose(Image.FLIP_TOP_BOTTOM) for v in images], mask.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return images, mask


class JointRandomHorizontalFlip:
    """Callable to randomly flip images and their mask left to right.
    """

    def __init__(self, p):
        """Creates an `JointRandomHorizontalFlip` instance.

        Args:
          p: the probability for flipping.
        """

        self.p = p

    def __call__(self, images, mask):
        """Randomly flips image and their mask left to right.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask flipped or none of them flipped.
        """

        if random.random() < self.p:
            return [v.transpose(Image.FLIP_LEFT_RIGHT) for v in images], mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return images, mask


class JointRandomRotation:
    """Callable to randomly rotate images and their mask.
    """

    def __init__(self, p, degree):
        """Creates an `JointRandomRotation` instance.

        Args:
          p: the probability for rotating.
        """

        self.p = p

        methods = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}

        if degree not in methods.keys():
            raise NotImplementedError("We only support multiple of 90 degree rotations for now")

        self.method = methods[degree]

    def __call__(self, images, mask):
        """Randomly rotates images and their mask.

        Args:
          images: the PIL.Image image to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask rotated or none of them rotated.
        """

        if random.random() < self.p:
            return [v.transpose(self.method) for v in images], mask.transpose(self.method)
        else:
            return images, mask
