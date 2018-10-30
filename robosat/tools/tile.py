import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import mercantile
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
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
    parser.add_argument("--no_edges", action="store_true", help="don't generate edges tiles (to avoid black margins)")
    parser.add_argument("--label_threshold", type=int, default=1, help="label value threshold")
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
        raster = rio.open(args.raster)
        warp_vrt = WarpedVRT(raster, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
        bounds = transform_bounds(*[raster.crs, "epsg:4326"] + list(raster.bounds))
        tiles = [mercantile.Tile(x=x, y=y, z=z) for x, y, z in mercantile.tiles(*bounds + (args.zoom,))]
    except:
        sys.exit("Error: Unable to load raster or deal with it's projection")

    if args.no_edges:
        edges_x = (min(tiles, key=lambda xy: xy[0])[0]), (max(tiles, key=lambda xy: xy[0])[0])
        edges_y = (min(tiles, key=lambda xy: xy[1])[1]), (max(tiles, key=lambda xy: xy[1])[1])
        tiles = [mercantile.Tile(x=x, y=y, z=z) for x, y, z in tiles if x not in edges_x and y not in edges_y]
        assert len(tiles), "Error: Nothing left to tile, once the edges removed"

    for tile in tqdm(tiles, desc="Tiling", unit="tile", ascii=True):

        os.makedirs(os.path.join(args.out, str(args.zoom), str(tile.x)), exist_ok=True)
        path = os.path.join(args.out, str(args.zoom), str(tile.x), str(tile.y))
        data = warp_vrt.read(
            out_shape=(len(raster.indexes), args.size, args.size),
            window=warp_vrt.window(*mercantile.xy_bounds(tile)),
            boundless=True,
        )
        C, W, H = data.shape

        if args.type == "label":
            assert C == 1, "Error: Label raster input should be 1 band"

            data[data < args.label_threshold] = 0
            data[data >= args.label_threshold] = 1

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
        leaflet(args.out, args.leaflet, tiles, ext)
