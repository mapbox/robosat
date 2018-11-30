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
    parser.add_argument("--path", type=str, help="path to user's extension modules dir")
    parser.add_argument("pbf", type=str, help="path to .osm.pbf base map")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):
    module_search_path = [args.path] if args.path else []
    module_search_path.append(os.path.join(Path(__file__).parent.parent, "osm"))
    modules = [(path, name) for path, name, _ in pkgutil.iter_modules(module_search_path) if name != "core"]
    if args.type not in [name for _, name in modules]:
        sys.exit("Unknown type, thoses available are {}".format([name for _, name in modules]))

    if args.path:
        sys.path.append(args.path)
        module = import_module(args.type)
    else:
        module = import_module("robosat.osm.{}".format(args.type))

    handler = getattr(module, "{}Handler".format(args.type.title()))()
    handler.apply_file(filename=args.pbf, locations=True)
    handler.save(args.out)
