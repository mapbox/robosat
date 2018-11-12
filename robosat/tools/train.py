import os
import sys
import argparse
import collections
from contextlib import contextmanager

from PIL import Image

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from tqdm import tqdm

from robosat.transforms import (
    JointCompose,
    JointTransform,
    JointResize,
    JointRandomFlipOrRotate,
    ImageToTensor,
    MaskToTensor,
)
from robosat.datasets import SlippyMapTilesConcatenation
from robosat.metrics import Metrics
from robosat.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from robosat.unet import UNet
from robosat.utils import plot
from robosat.config import load_config
from robosat.log import Log


@contextmanager
def no_grad():
    with torch.no_grad():
        yield


def add_parser(subparser):
    parser = subparser.add_parser(
        "train", help="trains model on dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True, help="path to model configuration file")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
    parser.add_argument("--resume", action="store_true", help="resume training (imply to provide a checkpoint)")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    os.makedirs(model["common"]["checkpoint"], exist_ok=True)
    log = Log(os.path.join(model["common"]["checkpoint"], "log"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        log.log("RoboSat - training on {} GPUs, with {} workers".format(torch.cuda.device_count(), args.workers))
    else:
        device = torch.device("cpu")
        log.log("RoboSat - training on CPU, with {} workers", format(args.workers))

    num_classes = len(dataset["common"]["classes"])
    num_channels = 0
    for channel in dataset["channels"]:
        num_channels += len(channel["bands"])
    pretrained = model["opt"]["pretrained"]
    net = DataParallel(UNet(num_classes, num_channels=num_channels, pretrained=pretrained)).to(device)

    if model["opt"]["loss"] in ("CrossEntropy", "mIoU", "Focal"):
        try:
            weight = torch.Tensor(dataset["weights"]["values"])
        except KeyError:
            sys.exit("Error: The loss function used, need dataset weights values")

    optimizer = Adam(net.parameters(), lr=model["opt"]["lr"], weight_decay=model["opt"]["decay"])

    resume = 0
    if args.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if torch.cuda.is_available() else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(args.checkpoint, map_location=map_location)
        net.load_state_dict(chkpt["state_dict"])
        log.log("Using checkpoint: {}".format(args.checkpoint))

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]

    if model["opt"]["loss"] == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "mIoU":
        criterion = mIoULoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Focal":
        criterion = FocalLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Lovasz":
        criterion = LovaszLoss2d().to(device)
    else:
        sys.exit("Error: Unknown [opt][loss] value !")

    train_loader, val_loader = get_dataset_loaders(model, dataset, args.workers)

    num_epochs = model["opt"]["epochs"]
    if resume >= num_epochs:
        sys.exit("Error: Epoch {} set in {} already reached by the checkpoint provided".format(num_epochs, args.model))

    history = collections.defaultdict(list)

    log.log("")
    log.log("--- Input tensor from Dataset: {} ---".format(dataset["common"]["dataset"]))
    num_channel = 1
    for channel in dataset["channels"]:
        for band in channel["bands"]:
            log.log("Channel {}:\t\t {}[band: {}]".format(num_channel, channel["sub"], band))
            num_channel += 1
    log.log("")
    log.log("--- Hyper Parameters ---")
    log.log("Batch Size:\t\t {}".format(model["common"]["batch_size"]))
    log.log("Image Size:\t\t {}".format(model["common"]["image_size"]))
    log.log("Data Augmentation:\t {}".format(model["opt"]["data_augmentation"]))
    log.log("Learning Rate:\t\t {}".format(model["opt"]["lr"]))
    log.log("Weight Decay:\t\t {}".format(model["opt"]["decay"]))
    log.log("Loss function:\t\t {}".format(model["opt"]["loss"]))
    log.log("ResNet pre-trained:\t {}".format(model["opt"]["pretrained"]))
    if "weight" in locals():
        log.log("Weights :\t\t {}".format(dataset["weights"]["values"]))
    log.log("")

    for epoch in range(resume, num_epochs):
        log.log("---")
        log.log("Epoch: {}/{}".format(epoch + 1, num_epochs))

        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        log.log(
            "Train    loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                train_hist["loss"],
                train_hist["miou"],
                dataset["common"]["classes"][1],
                train_hist["fg_iou"],
                train_hist["mcc"],
            )
        )

        for k, v in train_hist.items():
            history["train " + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        log.log(
            "Validate loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                val_hist["loss"], val_hist["miou"], dataset["common"]["classes"][1], val_hist["fg_iou"], val_hist["mcc"]
            )
        )

        for k, v in val_hist.items():
            history["val " + k].append(v)

        visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
        plot(os.path.join(model["common"]["checkpoint"], visual), history)

        checkpoint = "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, num_epochs)
        states = {"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(states, os.path.join(model["common"]["checkpoint"], checkpoint))


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()

    for images, masks, tiles in tqdm(loader, desc="Train", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


@no_grad()
def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images, masks, tiles in tqdm(loader, desc="Validate", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def get_dataset_loaders(model, dataset, workers):

    # Values computed on ImageNet DataSet
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = JointCompose(
        [
            JointResize(model["common"]["image_size"]),
            JointRandomFlipOrRotate(model["opt"]["data_augmentation"]),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    train_dataset = SlippyMapTilesConcatenation(
        os.path.join(dataset["common"]["dataset"], "training"),
        dataset["channels"],
        os.path.join(dataset["common"]["dataset"], "training", "labels"),
        joint_transform=transform,
    )

    val_dataset = SlippyMapTilesConcatenation(
        os.path.join(dataset["common"]["dataset"], "validation"),
        dataset["channels"],
        os.path.join(dataset["common"]["dataset"], "validation", "labels"),
        joint_transform=transform,
    )

    batch_size = model["common"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader
