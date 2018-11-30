import os
import io
import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torchvision.transforms import Compose, Normalize

import mercantile
import requests
import cv2
from PIL import Image
from flask import Flask, send_file, render_template, abort

from robosat.tiles import fetch_image
from robosat.unet import UNet
from robosat.config import load_config
from robosat.colors import make_palette
from robosat.transforms import ImageToTensor

"""
Simple tile server running a segmentation model on the fly.

Endpoints:
  /zoom/x/y.png  Segmentation mask PNG image for the corresponding tile

Note: proof of concept for quick visualization only; limitations:
  Needs to be single threaded, request runs prediction on the GPU (singleton); should be batch prediction
  Does not take surrounding tiles into account for prediction; border predictions do not have to match
  Downloads satellite images for each request; should request internal data or at least do some caching
"""

app = Flask(__name__)

session = None
predictor = None
tiles = None
token = None
size = None


@app.route("/")
def index():
    return render_template("map.html", token=token, size=size)


@app.route("/<int:z>/<int:x>/<int:y>.png")
def tile(z, x, y):

    # Todo: predictor should take care of zoom levels
    if z != 18:
        abort(404)

    tile = mercantile.Tile(x, y, z)

    url = tiles.format(x=tile.x, y=tile.y, z=tile.z)
    res = fetch_image(session, url)

    if not res:
        abort(500)

    image = cv2.imdecode(np.asarray(bytearray(res.read()), dtype=np.uint8), cv2.COLOR_BGR2RGB)

    mask = predictor.segment(image)

    return send_png(mask)


@app.after_request
def after_request(response):
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    return response


def add_parser(subparser):
    parser = subparser.add_parser(
        "serve",
        help="serves predicted masks with on-demand tileserver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="path to configuration file")

    parser.add_argument("--url", type=str, help="endpoint with {z}/{x}/{y} variables to fetch image tiles from")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("--tile_size", type=int, default=512, help="tile size for slippy map tiles")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host to serve on")
    parser.add_argument("--port", type=int, default=5000, help="port to serve on")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)

    global size
    size = args.tile_size

    global token
    token = os.getenv("MAPBOX_ACCESS_TOKEN")

    if not token:
        sys.exit("Error: map token needed visualizing results; export MAPBOX_ACCESS_TOKEN")

    global session
    session = requests.Session()

    global tiles
    tiles = args.url

    global predictor
    predictor = Predictor(args.checkpoint, config)

    app.run(host=args.host, port=args.port, threaded=False)


def send_png(image):
    output = io.BytesIO()
    image.save(output, format="png", optimize=True)
    output.seek(0)
    return send_file(output, mimetype="image/png")


class Predictor:
    def __init__(self, checkpoint, config):

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.checkpoint = checkpoint
        self.config = config

        self.net = self.net_from_chkpt_()

    def segment(self, image):
        # don't track tensors with autograd during prediction
        with torch.no_grad():
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

            transform = Compose([ImageToTensor(), Normalize(mean=mean, std=std)])
            image = transform(image)

            batch = image.unsqueeze(0).to(self.device)

            output = self.net(batch)

            output = output.cpu().data.numpy()
            output = output.squeeze(0)

            mask = output.argmax(axis=0).astype(np.uint8)

            mask = Image.fromarray(mask, mode="P")

            palette = make_palette(*self.config["common"]["colors"])
            mask.putpalette(palette)

            return mask

    def net_from_chkpt_(self):
        def map_location(storage, _):
            return storage.cuda() if self.cuda else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(self.checkpoint, map_location=map_location)

        num_classes = len(self.config["classes"]["titles"])

        net = UNet(num_classes).to(self.device)
        net = nn.DataParallel(net)

        if self.cuda:
            torch.backends.cudnn.benchmark = True

        net.load_state_dict(chkpt["state_dict"])
        net.eval()

        return net
