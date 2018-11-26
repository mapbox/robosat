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
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("masks", type=str, help="slippy map directory with segmentation masks")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)

    module_path_search = [os.path.join(Path(__file__).parent.parent, "features")]
    modules = [name for _, name, _ in pkgutil.iter_modules(module_path_search) if name != "core"]
    if args.type not in modules:
        sys.exit("Unknown type, thoses available are {}".format(modules))

    labels = dataset["common"]["classes"]
    if args.type not in labels:
        sys.exit("The type you asked is not consistent with yours classes in the dataset file provided.")
    index = labels.index(args.type)

    try:
        module = import_module("robosat.features." + args.type, package=__name__)
        handler = getattr(module, "{}Handler".format(args.type.title()))
    except:
        sys.exit("Something get wrong, unable to call {}Handler", args.type.title())

    for tile, path in tqdm(list(tiles_from_slippy_map(args.masks)), ascii=True, unit="mask"):
        image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        mask = (image == index).astype(np.uint8)
        handler().apply(tile, mask)

    handler().save(args.out)
