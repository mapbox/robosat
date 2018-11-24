import os
import sys
import math
import json
import torch
import argparse

from PIL import Image
from tqdm import tqdm
import numpy as np

from mercantile import feature

from robosat.colors import make_palette, complementary_palette
from robosat.tiles import tiles_from_slippy_map, tile_image
from robosat.config import load_config
from robosat.metrics import Metrics
from robosat.utils import web_ui
from robosat.log import Log


def add_parser(subparser):
    parser = subparser.add_parser(
        "compare", help="compare images and/or labels and masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", type=str, default="side", help="compare mode (e.g side, stack or list)")
    parser.add_argument("--images", type=str, nargs="+", help="slippy map images dirs to render (stack or side mode)")
    parser.add_argument("--ext", type=str, default="webp", help="file format to save images in (stack or side mode)")
    parser.add_argument("--labels", type=str, help="directory to read slippy map labels from (needed for QoD metric)")
    parser.add_argument("--masks", type=str, help="directory to read slippy map masks from (needed for QoD metric)")
    parser.add_argument("--dataset", type=str, help="path to dataset configuration file (needed for QoD metric)")
    parser.add_argument("--minimum_fg", type=float, default=0.0, help="skip tile if label foreground below, [0-100]")
    parser.add_argument("--maximum_fg", type=float, default=100.0, help="skip tile if label foreground above, [0-100]")
    parser.add_argument("--minimum_qod", type=float, default=0.0, help="skip tile if QoD metric below, [0-100]")
    parser.add_argument("--maximum_qod", type=float, default=100.0, help="skip tile if QoD metric above, [0-100]")
    parser.add_argument("--vertical", action="store_true", help="render vertical image aggregate, for side mode")
    parser.add_argument("--geojson", action="store_true", help="output geojson based, for list mode")
    parser.add_argument("--web_ui", type=str, help="web ui base url")
    parser.add_argument("out", type=str, help="directory or path (upon mode) to save output to")

    parser.set_defaults(func=main)


def compare(masks, labels, tile, classes):

    x, y, z = list(map(str, tile))
    label = np.array(Image.open(os.path.join(labels, z, x, "{}.png".format(y))))
    mask = np.array(Image.open(os.path.join(masks, z, x, "{}.png".format(y))))

    assert label.shape == mask.shape
    assert len(label.shape) == 2 and len(classes) == 2  # Still binary centric

    metrics = Metrics(classes)
    metrics.add(torch.from_numpy(label), torch.from_numpy(mask), is_prob=False)
    fg_iou = metrics.get_fg_iou()

    fg_ratio = 100 * max(np.sum(mask != 0), np.sum(label != 0)) / mask.size
    dist = 0.0 if math.isnan(fg_iou) else 1.0 - fg_iou

    qod = 100 - (dist * (math.log(fg_ratio + 1.0) + np.finfo(float).eps) * (100 / math.log(100)))
    qod = 0.0 if qod < 0.0 else qod  # Corner case prophilaxy

    return dist, fg_ratio, qod


def main(args):

    if not args.masks or not args.labels or not args.dataset:
        if args.mode == "list":
            sys.exit("Parameters masks, labels and dataset, are all mandatories in list mode.")
        if args.minimum_fg > 0 or args.maximum_fg < 100 or args.minimum_qod > 0 or args.maximum_qod < 100:
            sys.exit("Parameters masks, labels and dataset, are all mandatories in QoD filtering.")

    if args.images:
        tiles = [tile for tile, _ in tiles_from_slippy_map(args.images[0])]
        for image in args.images[1:]:
            assert sorted(tiles) == sorted([tile for tile, _ in tiles_from_slippy_map(image)]), "inconsistent coverages"

    if args.labels and args.masks:
        tiles_masks = [tile for tile, _ in tiles_from_slippy_map(args.masks)]
        tiles_labels = [tile for tile, _ in tiles_from_slippy_map(args.labels)]
        if args.images:
            assert sorted(tiles) == sorted(tiles_masks) == sorted(tiles_labels), "inconsistent coverages"
        else:
            assert sorted(tiles_masks) == sorted(tiles_labels), "inconsistent coverages"
            tiles = tiles_masks

    if args.mode == "list":
        out = open(args.out, mode="w")
        if args.geojson:
            out.write('{"type":"FeatureCollection","features":[')
            first = True

    tiles_compare = []
    for tile in tqdm(list(tiles), desc="Compare", unit="tile", ascii=True):

        x, y, z = list(map(str, tile))

        if args.masks and args.labels and args.dataset:
            classes = load_config(args.dataset)["common"]["classes"]
            dist, fg_ratio, qod = compare(args.masks, args.labels, tile, classes)
            if not args.minimum_fg <= fg_ratio <= args.maximum_fg or not args.minimum_qod <= qod <= args.maximum_qod:
                continue

        tiles_compare.append(tile)

        if args.mode == "side":

            for i, image in enumerate(args.images):
                img = tile_image(image, x, y, z)

                if i == 0:
                    side = np.zeros((img.shape[0], img.shape[1] * len(args.images), 3))
                    side = np.swapaxes(side, 0, 1) if args.vertical else side
                    image_shape = img.shape
                else:
                    assert image_shape == img.shape, "Unconsistent image size to compare"

                if args.vertical:
                    side[i * image_shape[0] : (i + 1) * image_shape[0], :, :] = img
                else:
                    side[:, i * image_shape[0] : (i + 1) * image_shape[0], :] = img

            os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
            side = Image.fromarray(np.uint8(side))
            side.save(os.path.join(args.out, z, x, "{}.{}".format(y, args.ext)), optimize=True)

        elif args.mode == "stack":

            for i, image in enumerate(args.images):
                img = tile_image(image, x, y, z)

                if i == 0:
                    image_shape = img.shape[0:2]
                    stack = img / len(args.images)
                else:
                    assert image_shape == img.shape[0:2], "Unconsistent image size to compare"
                    stack = stack + (img / len(args.images))

            os.makedirs(os.path.join(args.out, str(z), str(x)), exist_ok=True)
            stack = Image.fromarray(np.uint8(stack))
            stack.save(os.path.join(args.out, str(z), str(x), "{}.{}".format(y, args.ext)), optimize=True)

        elif args.mode == "list":
            if args.geojson:
                prop = '"properties":{{"x":{},"y":{},"z":{},"fg":{:.1f},"qod":{:.1f}}}'.format(x, y, z, fg_ratio, qod)
                geom = '"geometry":{}'.format(json.dumps(feature(tile, precision=6)["geometry"]))
                out.write('{}{{"type":"Feature",{},{}}}'.format("," if not first else "", geom, prop))
                first = False
            else:
                out.write("{},{},{}\t\t{:.1f}\t\t{:.1f}{}".format(x, y, z, fg_ratio, qod, os.linesep))

        else:
            sys.exit("Unkown mode, should be either: side, stack or list")

    if args.mode == "list":
        if args.geojson:
            out.write("]}")
        out.close()

    elif args.mode == "side" and args.web_ui:
        web_ui(args.out, args.web_ui, None, tiles_compare, args.ext, "compare.html")

    elif args.mode == "stack" and args.web_ui:
        tiles = [tile for tile, _ in tiles_from_slippy_map(args.images[0])]
        web_ui(args.out, args.web_ui, tiles, tiles_compare, args.ext, "leaflet.html")
