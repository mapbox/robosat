import unittest

import torch
from robosat.transforms import JointCompose, JointTransform, ImageToTensor, MaskToTensor
import mercantile

from robosat.datasets import SlippyMapTiles, SlippyMapTilesConcatenation


class TestSlippyMapTiles(unittest.TestCase):

    images = "tests/fixtures/images/"

    def test_len(self):
        dataset = SlippyMapTiles(TestSlippyMapTiles.images, "image")
        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        dataset = SlippyMapTiles(TestSlippyMapTiles.images, "image")
        image, tile = dataset[0]

        assert tile == mercantile.Tile(69105, 105093, 18)
        self.assertEqual(image.size, 512 * 512 * 3)
        self.assertEqual(image.shape, (512, 512, 3))

    def test_getitem_with_transform(self):
        # TODO
        pass


class TestSlippyMapTilesConcatenation(unittest.TestCase):
    def test_len(self):
        inputs = ["tests/fixtures/images/"]
        target = "tests/fixtures/labels/"

        transform = JointCompose([JointTransform(ImageToTensor(), MaskToTensor())])
        dataset = SlippyMapTilesConcatenation(inputs, target, transform)

        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        inputs = ["tests/fixtures/images/"]
        target = "tests/fixtures/labels/"

        transform = JointCompose([JointTransform(ImageToTensor(), MaskToTensor())])
        dataset = SlippyMapTilesConcatenation(inputs, target, transform)

        images, mask, tiles = dataset[0]
        self.assertEqual(tiles[0], mercantile.Tile(69105, 105093, 18))
        self.assertEqual(type(images), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)
