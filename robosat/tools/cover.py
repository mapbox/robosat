import os
import sys
import argparse
import csv
import json

from tqdm import tqdm
from mercantile import tiles
from supermercado import burntiles

from robosat.datasets import tiles_from_slippy_map


def add_parser(subparser):
    parser = subparser.add_parser(
        "cover",
        help="generates tiles covering, in csv format: X,Y,Z",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zoom", type=int, help="zoom level of tiles")
    parser.add_argument("--type", type=str, default="geojson", choices=["geojson", "bbox", "dir"], help="input type")
    help = "input value, upon type either: a geojson file path, a lat/lon bbox in ESPG:4326, or a slippymap dir path"
    parser.add_argument("input", type=str, help=help)
    parser.add_argument("out", type=str, help="path to csv file to generate")

    parser.set_defaults(func=main)


def main(args):

    if not args.zoom and args.type in ["geojson", "bbox"]:
        sys.exit("Zoom parameter is mandatory")

    cover = []

    if args.type == "geojson":
        with open(args.input) as f:
            features = json.load(f)

        for feature in tqdm(features["features"], ascii=True, unit="feature"):
            cover.extend(map(tuple, burntiles.burn([feature], args.zoom).tolist()))

        cover = list(set(cover))  # tiles can overlap for multiple features; unique tile ids

    elif args.type == "bbox":
        west, south, east, north = map(float, args.input.split(","))
        cover = tiles(west, south, east, north, args.zoom)

    elif args.type == "dir":
        cover = [tile for tile, _ in tiles_from_slippy_map(args.input)]

    if not os.path.isdir(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as fp:
        csv.writer(fp).writerows(cover)
