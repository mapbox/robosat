import os
import sys
import argparse
import pkgutil
from pathlib import Path
from importlib import import_module


def add_parser(subparser):
    parser = subparser.add_parser(
        "extract",
        help="extracts GeoJSON features from OpenStreetMap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--type", type=str, required=True, help="type of feature to extract")
    parser.add_argument("pbf", type=str, help="path to .osm.pbf base map")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):
    module_path_search = [os.path.join(Path(__file__).parent.parent, "osm")]
    modules = [name for _, name, _ in pkgutil.iter_modules(module_path_search) if name != "core"]
    if args.type not in modules:
        sys.exit("Unknown type, thoses available are {}".format(modules))

    try:
        module = import_module("robosat.osm." + args.type, package=__name__)
        handler = getattr(module, "{}Handler".format(args.type.title()))
        handler().apply_file(filename=args.pbf, locations=True)
    except:
        sys.exit("Something get wrong, unable to call {}Handler", args.type.title())

    handler().save(args.out)
