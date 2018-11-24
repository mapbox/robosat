"""Color handling, color maps, color palettes.
"""

import colorsys
import webcolors
import numpy as np


def make_palette(*colors):
    """Builds a PIL-compatible color palette from CSS3 color names, or hex values patterns as #RRGGBB

    Args:
      colors: variable number of color names.
    """

    assert 0 < len(colors) <= 256

    hexs = [webcolors.CSS3_NAMES_TO_HEX[color] if color[0] != "#" else color for color in colors]
    rgbs = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexs]

    return list(sum(rgbs, ()))


def continuous_palette_for_color(color, bins=256):
    """Creates a continuous color palette based on a single color.

    Args:
      color: the CSS3 color name or it's hex values as #RRGGBB, to create a continuous palette for.
      bins: the number of colors to create in the continuous palette.

    Returns:
      The continuous rgb color palette with 3*bins values represented as [r0,g0,b0,r1,g1,b1,..]
    """

    # A quick and dirty way to create a continuous color palette is to convert from the RGB color
    # space into the HSV color space and then only adapt the color's saturation (S component).

    hexs = webcolors.CSS3_NAMES_TO_HEX[color] if color[0] != "#" else color
    r, g, b = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexs]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    assert 0 < bins <= 256

    palette = []
    for i in range(bins):
        r, g, b = [int(v * 255) for v in colorsys.hsv_to_rgb(h, (1 / bins) * (i + 1), v)]
        palette.extend(r, g, b)

    return palette


def complementary_palette(palette):
    """Creates a PIL complementary colors palette based on an initial PIL palette"""

    comp_palette = []
    colors = [palette[i : i + 3] for i in range(0, len(palette), 3)]

    for color in colors:
        r, g, b = [v for v in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        comp_palette.extend(map(int, colorsys.hsv_to_rgb((h + 0.5) % 1, s, v)))

    return comp_palette
