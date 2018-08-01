import argparse

import torch
import torch.onnx
import torch.autograd

from robosat.config import load_config
from robosat.unet import UNet


def add_parser(subparser):
    parser = subparser.add_parser(
        "export", help="exports model in ONNX format", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--image_size", type=int, default=512, help="image size to use for model")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("model", type=str, help="path to save ONNX GraphProto .pb model to")

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)

    num_classes = len(dataset["common"]["classes"])
    net = UNet(num_classes)

    def map_location(storage, _):
        return storage.cpu()

    chkpt = torch.load(args.checkpoint, map_location=map_location)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(chkpt)

    # Todo: make input channels configurable, not hard-coded to three channels for RGB
    batch = torch.autograd.Variable(torch.randn(1, 3, args.image_size, args.image_size))

    torch.onnx.export(net, batch, args.model)
