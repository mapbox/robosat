import argparse
import random

from robosat.tiles import tiles_from_slippy_map, tiles_to_csv


def main(args):
    tiles = []
    for tile, path in tiles_from_slippy_map(args.dir):
        tiles.append(tile)

    if args.shuffle:
        random.shuffle(tiles)

    if args.count:
        tiles = tiles[: args.count]

    tiles_to_csv(sorted(tiles), args.out)


def add_parser(subparser):
    parser = subparser.add_parser(
        "csv",
        help="writes files in a slippy map directory to a file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dir", type=str, help="slippy map directory to read images from")
    parser.add_argument("out", type=str, help="path to write csv of all images in directory")
    parser.add_argument("--shuffle", type=bool, default=False, help="whether to shuffle the images")
    parser.add_argument("--count", type=int, default=None, help="number of images to write to csv")

    parser.set_defaults(func=main)
