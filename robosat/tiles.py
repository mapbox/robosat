"""Slippy Map Tiles.

The Slippy Map tile spec works with a directory structure of `z/x/y.png` where
- `z` is the zoom level
- `x` is the left / right index
- `y` is the top / bottom index

See: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""

import csv
import io
import os
from glob import glob

import cv2
from PIL import Image
import numpy as np

from rasterio.warp import transform
from rasterio.crs import CRS
import mercantile


def pixel_to_location(tile, dx, dy):
    """Converts a pixel in a tile to a coordinate.

    Args:
      tile: the mercantile tile to calculate the location in.
      dx: the relative x offset in range [0, 1].
      dy: the relative y offset in range [0, 1].

    Returns:
      The coordinate for the pixel in the tile.
    """

    assert 0 <= dx <= 1, "x offset is in [0, 1]"
    assert 0 <= dy <= 1, "y offset is in [0, 1]"

    west, south, east, north = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat


def fetch_image(session, url, timeout=10):
    """Fetches the image representation for a tile.

    Args:
      session: the HTTP session to fetch the image from.
      url: the tile imagery's url to fetch the image from.
      timeout: the HTTP timeout in seconds.

    Returns:
     The satellite imagery as bytes or None in case of error.
    """

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception:
        return None


def tiles_from_slippy_map(root):
    """Loads files from an on-disk slippy map directory structure.

    Args:
      root: the base directory with layout `z/x/y.*`.

    Yields:
      The mercantile tiles and file paths from the slippy map directory.
    """

    # The Python string functions (.isdigit, .isdecimal, etc.) handle
    # unicode codepoints; we only care about digits convertible to int
    def isdigit(v):
        try:
            _ = int(v)  # noqa: F841
            return True
        except ValueError:
            return False

    root = os.path.expanduser(root)
    for z in os.listdir(root):
        if not isdigit(z):
            continue

        for x in os.listdir(os.path.join(root, z)):
            if not isdigit(x):
                continue

            for name in os.listdir(os.path.join(root, z, x)):
                y = os.path.splitext(name)[0]

                if not isdigit(y):
                    continue

                tile = mercantile.Tile(x=int(x), y=int(y), z=int(z))
                path = os.path.join(root, z, x, name)
                yield tile, path


def tiles_from_csv(path):
    """Read tiles from a line-delimited csv file.

    Args:
      file: the path to read the csv file from.

    Yields:
      The mercantile tiles from the csv file.
    """

    path = os.path.expanduser(path)
    with open(path) as fp:
        reader = csv.reader(fp)

        for row in reader:
            if not row:
                continue

            yield mercantile.Tile(*map(int, row))


def tile_image(root, x, y, z):
    """Retrieves H,W,C numpy array, from a tile store and X,Y,Z coordinates, or `None`"""

    try:
        root = os.path.expanduser(root)
        path = glob(os.path.join(root, z, x, y) + "*")
        assert len(path) == 1
        img = np.array(Image.open(path[0]).convert("RGB"))
    except:
        return None

    return img


def adjacent_tile_image(tile, dx, dy, tiles):
    """Retrieves an adjacent tile image from a tile store.

    Args:
      tile: the original tile to get an adjacent tile image for.
      dx: the offset in tile x direction.
      dy: the offset in tile y direction.
      tiles: the tile store to get tiles from; must support `__getitem__` with tiles.

    Returns:
      The adjacent tile's image or `None` if it does not exist.
    """

    x, y, z = map(int, [tile.x, tile.y, tile.z])
    adjacent = mercantile.Tile(x=x + dx, y=y + dy, z=z)

    try:
        path = tiles[adjacent]
    except KeyError:
        return None

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def buffer_tile_image(tile, tiles, overlap, tile_size):
    """Buffers a tile image adding borders on all sides based on adjacent tiles.

    Args:
      tile: the tile to buffer.
      tiles: available tiles; must be a mapping of tiles to their filesystem paths.
      overlap: the tile border to add on every side; in pixel.
      tile_size: the tile size.

    Returns:
      The H,W,C numpy composite image containing the original tile plus tile overlap on all sides.
      It's size is `tile_size` + 2 * `overlap` pixel for each side.
    """

    assert 0 <= overlap <= tile_size, "Overlap value can't be either negative or bigger than tile_size"

    tiles = dict(tiles)
    x, y, z = map(int, [tile.x, tile.y, tile.z])

    # 3x3 matrix (upper, center, bottom) x (left, center, right)
    ul = adjacent_tile_image(tile, -1, -1, tiles)
    uc = adjacent_tile_image(tile, +0, -1, tiles)
    ur = adjacent_tile_image(tile, +1, -1, tiles)
    cl = adjacent_tile_image(tile, -1, +0, tiles)
    cc = adjacent_tile_image(tile, +0, +0, tiles)
    cr = adjacent_tile_image(tile, +1, +0, tiles)
    bl = adjacent_tile_image(tile, -1, +1, tiles)
    bc = adjacent_tile_image(tile, +0, +1, tiles)
    br = adjacent_tile_image(tile, +1, +1, tiles)

    ts = tile_size
    o = overlap
    oo = overlap * 2

    # Todo: instead of nodata we should probably mirror the center image
    img = np.zeros((ts + oo, ts + oo, 3)).astype(np.uint8)

    # fmt:off
    img[0:o,        0:o,        :] = ul[-o:ts, -o:ts, :] if ul is not None else np.zeros((o,   o, 3)).astype(np.uint8)
    img[0:o,        o:ts+o,     :] = uc[-o:ts,  0:ts, :] if uc is not None else np.zeros((o,  ts, 3)).astype(np.uint8)
    img[0:o,        ts+o:ts+oo, :] = ur[-o:ts,   0:o, :] if ur is not None else np.zeros((o,   o, 3)).astype(np.uint8)
    img[o:ts+o,     0:o,        :] = cl[0:ts,  -o:ts, :] if cl is not None else np.zeros((ts,  o, 3)).astype(np.uint8)
    img[o:ts+o,     o:ts+o,     :] = cc                  if cc is not None else np.zeros((ts, ts, 3)).astype(np.uint8)
    img[o:ts+o,     ts+o:ts+oo, :] = cr[0:ts,    0:o, :] if cr is not None else np.zeros((ts,  o, 3)).astype(np.uint8)
    img[ts+o:ts+oo, 0:o,        :] = bl[0:o,   -o:ts, :] if bl is not None else np.zeros((o,   o, 3)).astype(np.uint8)
    img[ts+o:ts+oo, o:ts+o,     :] = bc[0:o,    0:ts, :] if bc is not None else np.zeros((o,  ts, 3)).astype(np.uint8)
    img[ts+o:ts+oo, ts+o:ts+oo, :] = br[0:o,     0:o, :] if br is not None else np.zeros((o,   o, 3)).astype(np.uint8)
    # fmt:on

    return img
