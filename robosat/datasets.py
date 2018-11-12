"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""

import os
import sys
import torch
from PIL import Image
import torch.utils.data
import cv2
import numpy as np

from robosat.tiles import tiles_from_slippy_map, buffer_tile_image


# Single Slippy Map directory structure
class SlippyMapTiles(torch.utils.data.Dataset):
    """Dataset for images stored in slippy map format.
    """

    def __init__(self, root, mode, transform=None):
        super().__init__()

        self.tiles = []
        self.transform = transform

        self.tiles = [(tile, path) for tile, path in tiles_from_slippy_map(root)]
        self.tiles.sort(key=lambda tile: tile[0])
        self.mode = mode

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]

        if self.mode == "image":
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        elif self.mode == "multibands":
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
            if len(image.shape) == 3 and image.shape[2] >= 3:
                # FIXME Look twice to find an in-place way to perform a multiband BGR2RGB
                g = image[:, :, 0]
                image[:, :, 0] = image[:, :, 2]
                image[:, :, 2] = g
            else:
                image = image.reshape(image.shape[0], image.shape[1], 1)

        elif self.mode == "mask":
            image = np.array(Image.open(path).convert("P"))

        if self.transform is not None:
            image = self.transform(image)

        return image, tile


# Multiple Slippy Map directories.
class SlippyMapTilesConcatenation(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, input_path, channels, target, joint_transform=None):
        super().__init__()

        assert len(channels), "Channels configuration empty"

        self.inputs = dict()
        for channel in channels:
            try:
                name, band = channel.split(":")
                self.inputs[name] = SlippyMapTiles(os.path.join(input_path, name), mode="multibands")
            except:
                sys.exit("Channels configuration issue")

        self.target = SlippyMapTiles(target, mode="mask")
        self.channels = channels

        # No transformations in the `SlippyMapTiles` instead joint transformations in getitem
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):

        mask, tile = self.target[i]

        for channel in self.channels:
            try:
                name, band = channel.split(":")
                data, input_tile = self.inputs[name][i]
                assert input_tile == tile
                data = data[:, :, int(band) - 1].reshape(mask.shape[0], mask.shape[1], 1)
                tensor = np.concatenate((tensor, data), axis=2) if "tensor" in locals() else data
            except:
                sys.exit("Unable to concatenate input Tensor")

        if self.joint_transform is not None:
            tensor, mask = self.joint_transform(tensor, mask)

        return tensor, mask, tile


# Todo: once we have the SlippyMapDataset this dataset should wrap
# it adding buffer and unbuffer glue on top of the raw tile dataset.
class BufferedSlippyMapDirectory(torch.utils.data.Dataset):
    """Dataset for buffered slippy map tiles with overlap.
    """

    def __init__(self, root, transform=None, size=512, overlap=32):
        """
        Args:
          root: the slippy map directory root with a `z/x/y.png` sub-structure.
          transform: the transformation to run on the buffered tile.
          size: the Slippy Map tile size in pixels
          overlap: the tile border to add on every side; in pixel.

        Note:
          The overlap must not span multiple tiles.

          Use `unbuffer` to get back the original tile.
        """

        super().__init__()

        assert overlap >= 0
        assert size >= 256

        self.transform = transform
        self.size = size
        self.overlap = overlap
        self.tiles = list(tiles_from_slippy_map(root))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, path = self.tiles[i]
        image = np.array(buffer_tile_image(tile, self.tiles, overlap=self.overlap, tile_size=self.size))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.IntTensor([tile.x, tile.y, tile.z])

    def unbuffer(self, probs):
        """Removes borders from segmentation probabilities added to the original tile image.

        Args:
          probs: the segmentation probability mask to remove buffered borders.

        Returns:
          The probability mask with the original tile's dimensions without added overlap borders.
        """

        o = self.overlap
        _, x, y = probs.shape

        return probs[:, o : x - o, o : y - o]
