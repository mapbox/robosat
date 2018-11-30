import os
import sys
import time
import argparse
import concurrent.futures as futures

import requests
from PIL import Image
from tqdm import tqdm
from mercantile import xy_bounds

from robosat.tiles import tiles_from_csv, fetch_image
from robosat.utils import web_ui
from robosat.log import Log


def add_parser(subparser):
    parser = subparser.add_parser(
        "download", help="downloads images from a remote server", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "url", type=str, help="endpoint with {z}/{x}/{y} or {xmin},{ymin},{xmax},{ymax} variables to fetch image tiles"
    )
    parser.add_argument("--ext", type=str, default="webp", help="file format to save images in")
    parser.add_argument("--rate", type=int, default=10, help="rate limit in max. requests per second")
    parser.add_argument("--type", type=str, default="XYZ", choices=["XYZ", "WMS", "TMS"], help="service type to use")
    parser.add_argument("--timeout", type=int, default=10, help="server request timeout (in seconds)")
    parser.add_argument("--web_ui", type=str, help="web ui client base url")
    parser.add_argument("--web_ui_template", type=str, help="path to an alternate web ui template")
    parser.add_argument("tiles", type=str, help="path to .csv tiles file")
    parser.add_argument("out", type=str, help="path to slippy map directory for storing tiles")

    parser.set_defaults(func=main)


def main(args):
    tiles = list(tiles_from_csv(args.tiles))
    already_dl = 0
    dl = 0

    with requests.Session() as session:
        num_workers = args.rate

        os.makedirs(os.path.join(args.out), exist_ok=True)
        log = Log(os.path.join(args.out, "log"), out=sys.stderr)
        log.log("Begin download from {}".format(args.url))

        # tqdm has problems with concurrent.futures.ThreadPoolExecutor; explicitly call `.update`
        # https://github.com/tqdm/tqdm/issues/97
        progress = tqdm(total=len(tiles), ascii=True, unit="image")

        with futures.ThreadPoolExecutor(num_workers) as executor:

            def worker(tile):
                tick = time.monotonic()

                x, y, z = map(str, [tile.x, tile.y, tile.z])

                os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
                path = os.path.join(args.out, z, x, "{}.{}".format(y, args.ext))

                if os.path.isfile(path):
                    return tile, None, True

                if args.type == "XYZ":
                    url = args.url.format(x=tile.x, y=tile.y, z=tile.z)
                elif args.type == "TMS":
                    tile.y = (2 ** tile.z) - tile.y - 1
                    url = args.url.format(x=tile.x, y=tile.y, z=tile.z)
                elif args.type == "WMS":
                    xmin, ymin, xmax, ymax = xy_bounds(tile)
                    url = args.url.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                res = fetch_image(session, url, args.timeout)
                if not res:
                    return tile, url, False

                try:
                    image = Image.open(res)
                    image.save(path, optimize=True)
                except OSError:
                    return tile, url, False

                tock = time.monotonic()

                time_for_req = tock - tick
                time_per_worker = num_workers / args.rate

                if time_for_req < time_per_worker:
                    time.sleep(time_per_worker - time_for_req)

                progress.update()

                return tile, url, True

            for tile, url, ok in executor.map(worker, tiles):
                if url and ok:
                    dl += 1
                elif not url and ok:
                    already_dl += 1
                else:
                    log.log("Warning:\n {} failed, skipping.\n {}\n".format(tile, url))

    if already_dl:
        log.log("Notice:\n {} tiles were already downloaded previously, and so skipped now.".format(already_dl))
    if already_dl + dl == len(tiles):
        log.log(" Coverage is fully downloaded.")

    if args.web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        web_ui(args.out, args.web_ui, tiles, tiles, args.ext, template)
