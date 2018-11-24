import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

from robosat.datasets import BufferedSlippyMapDirectory
from robosat.tiles import tiles_from_slippy_map
from robosat.unet import UNet
from robosat.config import load_config
from robosat.colors import continuous_palette_for_color, make_palette
from robosat.transforms import ImageToTensor
from robosat.utils import web_ui


def add_parser(subparser):
    parser = subparser.add_parser(
        "predict",
        help="predicts probability masks for slippy map tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--batch_size", type=int, default=1, help="images per batch")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("--overlap", type=int, default=32, help="tile pixel overlap to predict on")
    parser.add_argument("--tile_size", type=int, required=True, help="tile size for slippy map tiles")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    parser.add_argument("tiles", type=str, help="directory to read slippy map image tiles from")
    parser.add_argument("probs", type=str, help="directory to save slippy map probability masks to")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--masks_output", action="store_true", help="output masks rather than probs")
    parser.add_argument("--web_ui", type=str, help="web ui base url")

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)
    num_classes = len(dataset["common"]["classes"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    def map_location(storage, _):
        return storage.cuda() if torch.cuda.is_available() else storage.cpu()

    # https://github.com/pytorch/pytorch/issues/7178
    chkpt = torch.load(args.checkpoint, map_location=map_location)

    net = UNet(num_classes).to(device)
    net = nn.DataParallel(net)

    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose([ImageToTensor(), Normalize(mean=mean, std=std)])

    directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=args.tile_size, overlap=args.overlap)
    loader = DataLoader(directory, batch_size=args.batch_size, num_workers=args.workers)

    if args.masks_output:
        palette = make_palette(dataset["common"]["colors"][0], dataset["common"]["colors"][1])
    else:
        palette = continuous_palette_for_color("pink", 256)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))

                # we predicted on buffered tiles; now get back probs for original image
                prob = directory.unbuffer(prob)

                assert prob.shape[0] == 2, "single channel requires binary model"
                assert np.allclose(np.sum(prob, axis=0), 1.0), "single channel requires probabilities to sum up to one"

                if args.masks_output:
                    image = np.around(prob[1:, :, :]).astype(np.uint8).squeeze()
                else:
                    image = (prob[1:, :, :] * 255).astype(np.uint8).squeeze()

                out = Image.fromarray(image, mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(args.probs, str(z), str(x)), exist_ok=True)
                path = os.path.join(args.probs, str(z), str(x), str(y) + ".png")

                out.save(path, optimize=True)

    if args.web_ui:
        tiles = [tile for tile, _ in tiles_from_slippy_map(args.tiles)]
        web_ui(args.probs, args.web_ui, tiles, tiles, "png", "leaflet.html")
