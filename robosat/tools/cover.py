import argparse
import csv
import json
import sys

from supermercado import burntiles
from mercantile import tiles
from tqdm import tqdm


def add_parser(subparser):
    parser = subparser.add_parser(
        "cover",
        help="generates tiles covering GeoJSON features or lat/lon Bbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--features", type=str, help="path to GeoJSON features")
    parser.add_argument("--bbox", type=str, help="bbox expressed in lat/lon (i.e EPSG:4326)")
    parser.add_argument("out", type=str, help="path to csv file to store tiles in")

    parser.set_defaults(func=main)


def main(args):

    cover = []

    if args.features:
        with open(args.features) as f:
            features = json.load(f)

        for feature in tqdm(features["features"], ascii=True, unit="feature"):
            cover.extend(map(tuple, burntiles.burn([feature], args.zoom).tolist()))

        # tiles can overlap for multiple features; unique tile ids
        cover = list(set(cover))

    elif args.bbox:
        west, south, east, north = map(float, args.bbox.split(","))
        cover = tiles(west, south, east, north, args.zoom)

    else:
        sys.exit("You have to provide either a GeoJson features file, or a lat/lon bbox")

    with open(args.out, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(cover)
