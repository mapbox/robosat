import os
import sys
import argparse
import shutil

from glob import glob
from tqdm import tqdm

from robosat.tiles import tiles_from_csv
from robosat.utils import web_ui
from robosat.log import Log


def add_parser(subparser):
    parser = subparser.add_parser(
        "subset",
        help="filter images in a slippy map directory using a csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dir", type=str, help="directory to read slippy map tiles from for filtering")
    parser.add_argument("cover", type=str, help="csv cover to filter tiles by")
    parser.add_argument("out", type=str, help="directory to save filtered tiles to")
    parser.add_argument("--move", action="store_true", help="move files from src to dst (rather than copy them)" )
    parser.add_argument("--web_ui", type=str, help="web ui base url")
    parser.add_argument("--web_ui_template", type=str, help="path to an alternate web ui template")

    parser.set_defaults(func=main)


def main(args):
    log = Log(os.path.join(args.out, "log"), out=sys.stderr)

    tiles = set(tiles_from_csv(args.cover))
    extension = ""

    for tile in tqdm(tiles, desc="Subset", unit="tiles", ascii=True):

        paths = glob(os.path.join(args.dir, str(tile.z), str(tile.x), "{}.*".format(tile.y)))
        if len(paths) != 1:
            log.log("Warning: {} skipped.".format(tile))
            continue
        src = paths[0]

        try:
            extension = os.path.splitext(src)[1][1:]
            dst = os.path.join(args.out, str(tile.z), str(tile.x), "{}.{}".format(tile.y, extension))
            if not os.path.isdir(os.path.join(args.out, str(tile.z), str(tile.x))):
                os.makedirs(os.path.join(args.out, str(tile.z), str(tile.x)), exist_ok=True)
            if args.move:
                assert(os.path.isfile(src))
                shutil.move(src, dst)
            else:
                shutil.copyfile(src, dst)
        except:
            sys.exit("Error: Unable to process {}".format(tile))

    if args.web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        web_ui(args.out, args.web_ui, tiles, tiles, extension, template)
