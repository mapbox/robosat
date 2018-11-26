import argparse

import os
import torch
import torch.onnx
import torch.autograd
import torch.nn as nn

from robosat.config import load_config
from robosat.unet import UNet


def add_parser(subparser):
    parser = subparser.add_parser(
        "export", help="exports or prunes model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--export_channels", type=int, help="export channels to use (keep the first ones)")
    parser.add_argument("--type", type=str, choices=["onnx", "pth"], default="onnx", help="output type")
    parser.add_argument("--image_size", type=int, default=512, help="image size to use for model")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("out", type=str, help="path to save export model to")

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)

    if args.type == "onnx":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Workaround: PyTorch ONNX, DataParallel with GPU issue, cf https://github.com/pytorch/pytorch/issues/5315

    num_classes = len(dataset["common"]["classes"])
    num_channels = len(dataset["common"]["channels"])
    export_channels = num_channels if not args.export_channels else args.export_channels
    assert num_channels >= export_channels, "Will be hard indeed, to export more channels than thoses dataset provide"

    def map_location(storage, _):
        return storage.cpu()

    net = UNet(num_classes, num_channels=num_channels).to("cpu")
    chkpt = torch.load(args.checkpoint, map_location=map_location)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])

    if export_channels < num_channels:
        weights = torch.zeros((64, export_channels, 7, 7))
        weights.data = net.module.resnet.conv1.weight.data[:, :export_channels, :, :]
        net.module.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.module.resnet.conv1.weight = nn.Parameter(weights)

    if args.type == "onnx":
        batch = torch.autograd.Variable(torch.randn(1, export_channels, args.image_size, args.image_size))
        torch.onnx.export(net, batch, args.out)

    elif args.type == "pth":
        states = {"epoch": chkpt["epoch"], "state_dict": net.state_dict(), "optimizer": chkpt["optimizer"]}
        torch.save(states, args.out)
