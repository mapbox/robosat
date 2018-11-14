import os
import sys
import math
import torch
import argparse

from PIL import Image
from tqdm import tqdm
import numpy as np

from robosat.colors import make_palette, complementary_palette
from robosat.tiles import tiles_from_slippy_map
from robosat.config import load_config
from robosat.metrics import Metrics
from robosat.utils import leaflet
from robosat.log import Log


def add_parser(subparser):
    parser = subparser.add_parser(
        "compare", help="compare images, labels and masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("out", type=str, help="directory to save output to (or path in list mode)")
    parser.add_argument("--images", type=str, help="directory to read slippy map images from")
    parser.add_argument("--labels", type=str, help="directory to read slippy map labels from")
    parser.add_argument("--masks", type=str, help="directory to read slippy map masks from")
    parser.add_argument("--dirs", type=str, nargs="+", help="slippy map directories to compares (in side mode)")
    parser.add_argument("--mode", type=str, default="side", help="compare mode (e.g side, diff or list)")
    parser.add_argument("--dataset", type=str, help="path to dataset configuration file, (for diff and list modes)")
    parser.add_argument("--leaflet", type=str, help="leaflet client base url")
    parser.add_argument("--minimum_fg", type=float, default=0.0, help="skip if foreground ratio below, [0-100]")
    parser.add_argument("--minimum_qod", type=float, default=0.0, help="redshift tile if QoD below, [0-100]")

    parser.set_defaults(func=main)


def compare(mask, label, classes):

    # TODO: Still binary class centric

    metrics = Metrics(classes)
    metrics.add(torch.from_numpy(label), torch.from_numpy(mask), is_prob=False)
    fg_iou = metrics.get_fg_iou()

    fg_ratio = 100 * max(np.sum(mask != 0), np.sum(label != 0)) / mask.size
    dist = 0.0 if math.isnan(fg_iou) else 1.0 - fg_iou

    qod = 100 - (dist * (math.log(fg_ratio + 1.0) + np.finfo(float).eps) * (100 / math.log(100)))
    qod = 0.0 if qod < 0.0 else qod  # Corner case prophilaxy

    return dist, fg_ratio, qod


def main(args):

    if args.mode == "side":
        if not args.dirs or len(args.dirs) < 2:
            sys.exit("Error: In side mode, you must provide at least two directories")

        tiles = tiles_from_slippy_map(args.dirs[0])

        for tile, path in tqdm(list(tiles), desc="Compare", unit="image", ascii=True):

            width, heigh = Image.open(path).size
            side = Image.new(mode="RGB", size=(len(args.dirs) * width, height))

            x, y, z = list(map(str, tile))

            for i, dir in enumerate(args.dirs):
                try:
                    img = Image.open(os.path.join(dir, z, x, "{}.png".format(y)))
                except:
                    img = Image.open(os.path.join(dir, z, x, "{}.webp".format(y))).convert("RGB")

                assert image.size == img.size
                side.paste(img, box=(i * width, 0))

            os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
            path = os.path.join(args.out, z, x, "{}.webp".format(y))
            side.save(path, optimize=True)

    elif args.mode == "diff":
        if not args.images or not args.labels or not args.masks or not args.dataset:
            sys.exit("Error: in diff mode, you must provide images, labels and masks directories, and dataset path")

        dataset = load_config(args.dataset)
        classes = dataset["common"]["classes"]
        colors = dataset["common"]["colors"]
        assert len(classes) == len(colors), "classes and colors coincide"
        assert len(colors) == 2, "only binary models supported right now"

        palette_mask = make_palette(colors[0], colors[1])
        palette_label = complementary_palette(palette_mask)

        images = tiles_from_slippy_map(args.images)

        for tile, path in tqdm(list(images), desc="Compare", unit="image", ascii=True):
            x, y, z = list(map(str, tile))
            os.makedirs(os.path.join(args.out, str(z), str(x)), exist_ok=True)

            image = Image.open(path).convert("RGB")
            label = Image.open(os.path.join(args.labels, z, x, "{}.png".format(y)))
            mask = Image.open(os.path.join(args.masks, z, x, "{}.png".format(y)))

            assert image.size == label.size == mask.size
            assert label.getbands() == mask.getbands() == tuple("P")

            dist, fg_ratio, qod = compare(np.array(mask), np.array(label), classes)

            image = np.array(image) * 0.7
            if args.minimum_fg < fg_ratio and qod < args.minimum_qod:
                image += np.array(Image.new("RGB", label.size, (int(255 * 0.3), 0, 0)))

            mask.putpalette(palette_mask)
            label.putpalette(palette_label)
            mask = np.array(mask.convert("RGB"))
            label = np.array(label.convert("RGB"))

            diff = Image.fromarray(np.uint8((image + mask + label) / 3.0))
            diff.save(os.path.join(args.out, str(z), str(x), "{}.webp".format(y)), optimize=True)

        if args.leaflet:
            tiles = [tile for tile, _ in tiles_from_slippy_map(args.images)]
            leaflet(args.out, args.leaflet, tiles, "webp")

    elif args.mode == "list":
        if not args.labels or not args.masks or not args.dataset:
            sys.exit("In list mode, you must provide labels and masks directories, and dataset path")

        dataset = load_config(args.dataset)
        masks = tiles_from_slippy_map(args.masks)
        os.makedirs(os.path.basename(args.out), exist_ok=True)
        log = Log(args.out, out=None, mode="w")

        for tile, path in tqdm(list(masks), desc="Compare", unit="image", ascii=True):
            x, y, z = list(map(str, tile))
            mask = Image.open(os.path.join(args.masks, z, x, "{}.png".format(y)))
            label = Image.open(os.path.join(args.labels, z, x, "{}.png".format(y)))

            assert label.size == mask.size
            assert label.getbands() == mask.getbands() == tuple("P")

            dist, fg_ratio, qod = compare(np.array(mask), np.array(label), dataset["common"]["classes"])
            if args.minimum_fg < fg_ratio and qod < args.minimum_qod:
                log.log("{},{},{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(x, y, z, dist, fg_ratio, qod))
