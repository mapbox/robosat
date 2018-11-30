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
        path   = "tests/fixtures"
        target = "tests/fixtures/labels/"
        channels = [{"sub": "images", "bands": [1, 2, 3]}]

        transform = JointCompose([JointTransform(ImageToTensor(), MaskToTensor())])
        dataset = SlippyMapTilesConcatenation(path, channels, target, transform)

        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        path   = "tests/fixtures"
        target = "tests/fixtures/labels/"
        channels = [{"sub": "images", "bands": [1, 2, 3]}]

        transform = JointCompose([JointTransform(ImageToTensor(), MaskToTensor())])
        dataset = SlippyMapTilesConcatenation(path, channels, target, transform)

        images, mask, tiles = dataset[0]
        self.assertEqual(tiles, mercantile.Tile(69105, 105093, 18))
        self.assertEqual(type(images), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)
