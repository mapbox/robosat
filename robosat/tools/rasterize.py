import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles
from shapely.geometry import mapping

from robosat.config import load_config
from robosat.colors import make_palette, complementary_palette
from robosat.tiles import tiles_from_csv
from robosat.utils import web_ui
from robosat.log import Log


def add_parser(subparser):
    parser = subparser.add_parser(
        "rasterize", help="rasterize features to label masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("features", type=str, nargs="+", help="path to GeoJSON features file")
    parser.add_argument("cover", type=str, help="path to csv tiles cover file")
    parser.add_argument("out", type=str, help="directory to write converted images")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--size", type=int, default=512, help="size of rasterized image tiles in pixels")
    parser.add_argument("--web_ui", type=str, help="web ui client base url")

    parser.set_defaults(func=main)


def feature_to_mercator(feature):
    """Convert polygon feature coords to 3857.

    Args:
      feature: geojson feature to convert to mercator geometry.
    """
    # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe

    # FIXME: We assume that GeoJSON input coordinates can't be anything else than EPSG:4326
    if feature["geometry"]["type"] == "Polygon":
        xys = (zip(*ring) for ring in feature["geometry"]["coordinates"])
        xys = (list(zip(*transform(CRS.from_epsg(4326), CRS.from_epsg(3857), *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}


def burn(tile, features, size, burn_value=1):
    """Burn tile with features.

    Args:
      tile: the mercantile tile to burn.
      features: the geojson features to burn.
      size: the size of burned image.
      burn_value: the value you want in the output raster where a shape exists

    Returns:
      image: rasterized file of size with features burned.
    """

    shapes = ((geometry, burn_value) for feature in features for geometry in feature_to_mercator(feature))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, size, size)

    result = rasterize(shapes, out_shape=(size, size), transform=transform)
    return Image.fromarray(result, mode="P")


def main(args):
    dataset = load_config(args.dataset)

    classes = dataset["common"]["classes"]
    colors = dataset["common"]["colors"]
    assert len(classes) == len(colors), "classes and colors coincide"
    assert len(colors) == 2, "only binary models supported right now"

    os.makedirs(args.out, exist_ok=True)

    # We can only rasterize all tiles at a single zoom.
    assert all(tile.z == args.zoom for tile in tiles_from_csv(args.cover))

    # Find all tiles the features cover and make a map object for quick lookup.
    feature_map = collections.defaultdict(list)
    log = Log(os.path.join(args.out, "log"), out=sys.stderr)

    def parse_polygon(feature_map, polygon, i):

        try:
            for i, ring in enumerate(polygon["coordinates"]):  # GeoJSON coordinates could be N dimensionals
                polygon["coordinates"][i] = [[x, y] for point in ring for x, y in zip([point[0]], [point[1]])]

            for tile in burntiles.burn([{"type": "feature", "geometry": polygon}], zoom=args.zoom):
                feature_map[mercantile.Tile(*tile)].append({"type": "feature", "geometry": polygon})

        except ValueError as e:
            log.log("Warning: invalid feature {}, skipping".format(i))

        return feature_map

    def parse_geometry(feature_map, geometry, i):

        if geometry["type"] == "Polygon":
            feature_map = parse_polygon(feature_map, geometry, i)

        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                feature_map = parse_polygon(feature_map, {"type": "Polygon", "coordinates": polygon}, i)
        else:
            log.log("Notice: {} is a non surfacic geometry type, skipping feature {}".format(geometry["type"], i))

        return feature_map


    for feature in args.features:
        with open(feature) as f:
            fc = json.load(f)
            for i, feature in enumerate(tqdm(fc["features"], ascii=True, unit="feature")):

                if feature["geometry"]["type"] == "GeometryCollection":
                    for geometry in feature["geometry"]["geometries"]:
                        feature_map = parse_geometry(feature_map, geometry, i)
                else:
                    feature_map = parse_geometry(feature_map, feature["geometry"], i)

    # Burn features to tiles and write to a slippy map directory.
    for tile in tqdm(list(tiles_from_csv(args.cover)), ascii=True, unit="tile"):
        if tile in feature_map:
            out = burn(tile, feature_map[tile], args.size)
        else:
            out = Image.fromarray(np.zeros(shape=(args.size, args.size)).astype(int), mode="P")

        out_path = os.path.join(args.out, str(tile.z), str(tile.x))
        os.makedirs(out_path, exist_ok=True)

        out.putpalette(complementary_palette(make_palette(colors[0], colors[1])))
        out.save(os.path.join(out_path, "{}.png".format(tile.y)), optimize=True)

    if args.web_ui:
        tiles = [tile for tile in tiles_from_csv(args.cover)]
        web_ui(args.out, args.web_ui, tiles, tiles, "png", "leaflet.html")
