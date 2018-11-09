import os
import sys
import math
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import mercantile

from rasterio import open as rasterio_open
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, calculate_default_transform
from rasterio.transform import from_bounds

from robosat.config import load_config
from robosat.colors import make_palette
from robosat.utils import leaflet


def add_parser(subparser):
    parser = subparser.add_parser(
        "tile", help="tile a raster image or label", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("raster", type=str, help="path to the raster to tile")
    parser.add_argument("out", type=str, help="directory to write tiles")
    parser.add_argument("--size", type=int, default=512, help="size of tiles side in pixels")
    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--type", type=str, default="image", help="image or label tiling")
    parser.add_argument("--dataset", type=str, help="path to dataset configuration file, mandatory for label tiling")
    parser.add_argument("--no_data", type=int, help="color considered as no data [0-255]. Skip related tile")
    parser.add_argument("--leaflet", type=str, help="leaflet client base url")

    parser.set_defaults(func=main)


def main(args):

    if args.type == "label":
        try:
            dataset = load_config(args.dataset)
        except:
            sys.exit("Error: Unable to load DataSet config file")

        classes = dataset["common"]["classes"]
        colors = dataset["common"]["colors"]
        assert len(classes) == len(colors), "classes and colors coincide"
        assert len(colors) == 2, "only binary models supported right now"

    try:
        os.environ["GDAL_CACHEMAX"] = "50%"  # rasterio Env don't (yet) handle % settings
        raster = rasterio_open(args.raster)
        w, s, e, n = bounds = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds)
        transform, _, _ = calculate_default_transform(raster.crs, "EPSG:3857", raster.width, raster.height, *bounds)
    except:
        sys.exit("Error: Unable to load raster or deal with it's projection")

    tiles = [mercantile.Tile(x=x, y=y, z=z) for x, y, z in mercantile.tiles(w, s, e, n, args.zoom)]
    tiles_nodata = []

    for tile in tqdm(tiles, desc="Tiling", unit="tile", ascii=True):

        w, s, e, n = tile_bounds = mercantile.xy_bounds(tile)

        # Inspired by Rio-Tiler, cf: https://github.com/mapbox/rio-tiler/pull/45
        warp_vrt = WarpedVRT(
            raster,
            crs="EPSG:3857",
            resampling=Resampling.bilinear,
            add_alpha=False,
            transform=from_bounds(*tile_bounds, args.size, args.size),
            width=math.ceil((e - w) / transform.a),
            height=math.ceil((s - n) / transform.e),
        )
        data = warp_vrt.read(out_shape=(len(raster.indexes), args.size, args.size), window=warp_vrt.window(w, s, e, n))

        # If no_data is set, remove all tiles with at least one whole border filled only with no_data (on all bands)
        if type(args.no_data) is not None and (
            np.all(data[:, 0, :] == args.no_data)
            or np.all(data[:, -1, :] == args.no_data)
            or np.all(data[:, :, 0] == args.no_data)
            or np.all(data[:, :, -1] == args.no_data)
        ):
            tiles_nodata.append(tile)
            continue

        C, W, H = data.shape

        os.makedirs(os.path.join(args.out, str(args.zoom), str(tile.x)), exist_ok=True)
        path = os.path.join(args.out, str(args.zoom), str(tile.x), str(tile.y))

        if args.type == "label":
            assert C == 1, "Error: Label raster input should be 1 band"

            ext = "png"
            img = Image.fromarray(np.squeeze(data, axis=0), mode="P")
            img.putpalette(make_palette(colors[0], colors[1]))
            img.save("{}.{}".format(path, ext), optimize=True)

        elif args.type == "image":
            assert C == 1 or C == 3, "Error: Image raster input should be either 1 or 3 bands"

            # GeoTiff could be 16 or 32bits
            if data.dtype == "uint16":
                data = np.uint8(data / 256)
            elif data.dtype == "uint32":
                data = np.uint8(data / (256 * 256))

            if C == 1:
                ext = "png"
                Image.fromarray(np.squeeze(data, axis=0), mode="L").save("{}.{}".format(path, ext), optimize=True)
            elif C == 3:
                ext = "webp"
                Image.fromarray(np.moveaxis(data, 0, 2), mode="RGB").save("{}.{}".format(path, ext), optimize=True)

        else:
            sys.exit("Error: Unknown type, should be either 'image' or 'label'")

    if args.leaflet:
        leaflet(args.out, args.leaflet, [tile for tile in tiles if tile not in tiles_nodata], ext)
