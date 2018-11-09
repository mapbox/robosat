"""Color handling, color maps, color palettes.
"""

import colorsys
import numpy as np

from enum import Enum, unique


# Todo: user should be able to bring her own color palette.
# Functions need to account for that and not use one palette.


def _rgb(v):
    r, g, b = v[1:3], v[3:5], v[5:7]
    return int(r, 16), int(g, 16), int(b, 16)


@unique
class Mapbox(Enum):
    """Mapbox-themed colors.

    See: https://www.mapbox.com/base/styling/color/
    """

    dark = _rgb("#404040")
    gray = _rgb("#eeeeee")
    light = _rgb("#f8f8f8")
    white = _rgb("#ffffff")
    cyan = _rgb("#3bb2d0")
    blue = _rgb("#3887be")
    bluedark = _rgb("#223b53")
    denim = _rgb("#50667f")
    navy = _rgb("#28353d")
    navydark = _rgb("#222b30")
    purple = _rgb("#8a8acb")
    teal = _rgb("#41afa5")
    green = _rgb("#56b881")
    yellow = _rgb("#f1f075")
    mustard = _rgb("#fbb03b")
    orange = _rgb("#f9886c")
    red = _rgb("#e55e5e")
    pink = _rgb("#ed6498")


def make_palette(*colors, colormap=False):
    """Builds a PIL-compatible color palette from color names.

    Args:
      colors: variable number of color names.
    """

    assert 0 < len(colors) <= 256
    rgbs = [Mapbox[color].value for color in colors]

    if colormap:
        palette = np.zeros((256, 1, 3), np.uint8)
        for i in range(len(colors)):
            r, g, b = rgbs[i]
            palette[i, 0, :] = b, g, r
    else:
        palette = list(sum(rgbs, ()))

    return palette


def color_string_to_rgb(color):
    """Convert color string to a list of RBG integers.

    Args:
      color: the string color value for example "250,0,0"

    Returns:
      color: as a list of RGB integers for example [250,0,0]
    """

    return [*map(int, color.split(","))]


def continuous_palette_for_color(color, bins=256, colormap=False):
    """Creates a continuous color palette based on a single color.

    Args:
      color: the rgb color tuple to create a continuous palette for.
      bins: the number of colors to create in the continuous palette.

    Returns:
      The continuous rgb color palette with 3*bins values represented as [r0,g0,b0,r1,g1,b1,..]
    """

    # A quick and dirty way to create a continuous color palette is to convert from the RGB color
    # space into the HSV color space and then only adapt the color's saturation (S component).

    r, g, b = [v / 255 for v in Mapbox[color].value]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    assert 0 < bins <= 256
    palette = np.zeros((256, 1, 3), np.uint8) if colormap else list()

    for i in range(bins):
        r, g, b = [int(v * 255) for v in colorsys.hsv_to_rgb(h, (1 / bins) * (i + 1), v)]
        if colormap:
            palette[i, 0, :] = b, g, r
        else:
            palette.extend(r, g, b)

    return palette


def complementary_palette(palette):

    comp_palette = []
    colors = [palette[i:i+3] for i in range(0, len(palette), 3)]

    for color in colors:
        r, g, b = [v for v in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        comp_palette.extend(map(int, colorsys.hsv_to_rgb((h + 0.5) % 1, s, v)))

    return comp_palette
