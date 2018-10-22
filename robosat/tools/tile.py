import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import mercantile
from rio_tiler import main as tiler
from robosat.config import load_config
from robosat.colors import make_palette


def add_parser(subparser):
    parser = subparser.add_parser(
        "tile", help="tile a raster image", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("raster", type=str, help="path to the raster to tile")
    parser.add_argument("out", type=str, help="directory to write tiles")
    parser.add_argument("--size", type=int, default=512, help="size of tiles side in pixels")
    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--type", type=str, default="image", help="image or label tiling")
    parser.add_argument("--dataset", type=str, help="path to dataset configuration file, needed for label tiling")
    parser.add_argument("--no_edges", type=bool, help="don't generate edges tiles")

    parser.set_defaults(func=main)


def main(args):

    if args.type == "label":
        try:
            dataset = load_config(args.dataset)
        except:
            print("Error: Unable to load DataSet config file", file=sys.stderr)
            sys.exit()

        classes = dataset["common"]["classes"]
        colors = dataset["common"]["colors"]
        assert len(classes) == len(colors), "classes and colors coincide"
        assert len(colors) == 2, "only binary models supported right now"

    bounds = tiler.bounds(args.raster)["bounds"]
    tiles = [[x, y] for x, y, z in mercantile.tiles(*bounds + [[args.zoom]])]

    if args.no_edges:
        edges_x = (min(tiles, key=lambda xy: xy[0])[0]), (max(tiles, key=lambda xy: xy[0])[0])
        edges_y = (min(tiles, key=lambda xy: xy[1])[1]), (max(tiles, key=lambda xy: xy[1])[1])
        tiles = [[x, y] for x, y in tiles if x not in edges_x and y not in edges_y]
        assert len(tiles), "Error: Nothing left to tile, once remove the edges"

    for x, y in tqdm(tiles, desc="Tiling", unit="tile", ascii=True):

        os.makedirs(os.path.join(args.out, str(args.zoom), str(x)), exist_ok=True)
        path = os.path.join(args.out, str(args.zoom), str(x), str(y))
        data = tiler.tile(args.raster, x, y, args.zoom, tilesize=args.size)[0]

        C, W, H = data.shape

        if args.type == "label":
            assert C == 1, "Error: Label raster input should be 1 band"

            img = Image.fromarray(np.squeeze(data, axis=0), mode="P")
            img.putpalette(make_palette(colors[0], colors[1]))
            img.save(path + ".png", optimize=True)

        elif args.type == "image":
            assert C == 1 or C == 3, "Error: Image raster input should be either 1 or 3 bands"

            # GeoTiff could be 16 or 32bits
            if data.dtype == "uint16":
                data = np.uint8(data / 256)
            elif data.dtype == "uint32":
                data = np.uint8(data / (256 * 256))

            if C == 1:
                Image.fromarray(np.squeeze(data, axis=0), mode="L").save(path + ".png", optimize=True)
            elif C == 3:
                Image.fromarray(data, mode="RGB").save(path + ".webp", optimize=True)
