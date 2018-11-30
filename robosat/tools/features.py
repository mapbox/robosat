import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import pkgutil
from pathlib import Path
from importlib import import_module

from robosat.config import load_config
from robosat.tiles import tiles_from_slippy_map


def add_parser(subparser):
    parser = subparser.add_parser(
        "features",
        help="extracts simplified GeoJSON features from segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--type", type=str, required=True, help="type of feature to extract")
    parser.add_argument("--config", type=str, required=True, help="path to configuration file")
    parser.add_argument("--path", type=str, help="path to user's extension modules dir")
    parser.add_argument("masks", type=str, help="slippy map directory with segmentation masks")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):

    module_search_path = [args.path] if args.path else []
    module_search_path.append(os.path.join(Path(__file__).parent.parent, "features"))
    modules = [(path, name) for path, name, _ in pkgutil.iter_modules(module_search_path) if name != "core"]
    if args.type not in [name for _, name in modules]:
        sys.exit("Unknown type, thoses available are {}".format([name for _, name in modules]))

    config = load_config(args.config)
    labels = config["classes"]["titles"]
    if args.type not in labels:
        sys.exit("The type you asked is not consistent with yours classes in the config file provided.")
    index = labels.index(args.type)

    if args.path:
        sys.path.append(args.path)
        module = import_module(args.type)
    else:
        module = import_module("robosat.features.{}".format(args.type))

    handler = getattr(module, "{}Handler".format(args.type.title()))()

    for tile, path in tqdm(list(tiles_from_slippy_map(args.masks)), ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        mask = (image == index).astype(np.uint8)
        handler.apply(tile, mask)

    handler.save(args.out)
