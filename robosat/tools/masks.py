import os
import sys
import argparse

import numpy as np

from tqdm import tqdm
from PIL import Image

from robosat.tiles import tiles_from_slippy_map
from robosat.colors import make_palette
from robosat.utils import web_ui


def add_parser(subparser):
    parser = subparser.add_parser(
        "masks",
        help="compute masks from prediction probabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="path to configuration file")
    parser.add_argument("--weights", type=float, nargs="+", help="weights for weighted average soft-voting")
    parser.add_argument("--web_ui", type=str, help="web ui client base url")
    parser.add_argument("--web_ui_template", type=str, help="path to an alternate web ui template")
    parser.add_argument("masks", type=str, help="slippy map directory to save masks to")
    parser.add_argument("probs", type=str, nargs="+", help="slippy map directories with class probabilities")

    parser.set_defaults(func=main)


def main(args):
    if args.weights and len(args.probs) != len(args.weights):
        sys.exit("Error: number of slippy map directories and weights must be the same")

    tilesets = map(tiles_from_slippy_map, args.probs)

    for tileset in tqdm(list(zip(*tilesets)), desc="Masks", unit="tile", ascii=True):
        tiles = [tile for tile, _ in tileset]
        paths = [path for _, path in tileset]

        assert len(set(tiles)), "tilesets in sync"
        x, y, z = tiles[0]

        # Un-quantize the probabilities in [0,255] to floating point values in [0,1]
        anchors = np.linspace(0, 1, 256)

        def load(path):
            # Note: assumes binary case and probability sums up to one.
            # Needs to be in sync with how we store them in prediction.

            quantized = np.array(Image.open(path).convert("P"))

            # (512, 512, 1) -> (1, 512, 512)
            foreground = np.rollaxis(np.expand_dims(anchors[quantized], axis=0), axis=0)
            background = np.rollaxis(1. - foreground, axis=0)

            # (1, 512, 512) + (1, 512, 512) -> (2, 512, 512)
            return np.concatenate((background, foreground), axis=0)

        probs = [load(path) for path in paths]

        mask = softvote(probs, axis=0, weights=args.weights)
        mask = mask.astype(np.uint8)

        config = load_config(args.config)
        palette = make_palette(config["classes"]["colors"][0], config["classes"]["colors"][1])

        out = Image.fromarray(mask, mode="P")
        out.putpalette(palette)

        os.makedirs(os.path.join(args.masks, str(z), str(x)), exist_ok=True)

        path = os.path.join(args.masks, str(z), str(x), str(y) + ".png")
        out.save(path, optimize=True)

    if args.web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        tiles = [tile for tile, _ in list(tiles_from_slippy_map(args.probs[0]))]
        web_ui(args.masks, args.web_ui, tiles, tiles, "png", template)


def softvote(probs, axis=0, weights=None):
    """Weighted average soft-voting to transform class probabilities into class indices.

    Args:
      probs: array-like probabilities to average.
      axis: axis or axes along which to soft-vote.
      weights: array-like for weighting probabilities.

    Notes:
      See http://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    """

    return np.argmax(np.average(probs, axis=axis, weights=weights), axis=axis)
