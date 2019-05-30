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
from torchvision.transforms import Resize, CenterCrop, Normalize

from tqdm import tqdm

from robosat.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
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
    parser.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    device = torch.device("cuda" if model["common"]["cuda"] else "cpu")

    if model["common"]["cuda"] and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    os.makedirs(model["common"]["checkpoint"], exist_ok=True)

    num_classes = len(dataset["common"]["classes"])
    net = UNet(num_classes)
    net = DataParallel(net)
    net = net.to(device)

    if model["common"]["cuda"]:
        torch.backends.cudnn.benchmark = True

    try:
        weight = torch.Tensor(dataset["weights"]["values"])
    except KeyError:
        if model["opt"]["loss"] in ("CrossEntropy", "mIoU", "Focal"):
            sys.exit("Error: The loss function used, need dataset weights values")

    optimizer = Adam(net.parameters(), lr=model["opt"]["lr"])

    resume = 0
    if args.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if model["common"]["cuda"] else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(args.checkpoint, map_location=map_location)
        net.load_state_dict(chkpt["state_dict"])

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
    log = Log(os.path.join(model["common"]["checkpoint"], "log"))

    log.log("--- Hyper Parameters on Dataset: {} ---".format(dataset["common"]["dataset"]))
    log.log("Batch Size:\t {}".format(model["common"]["batch_size"]))
    log.log("Image Size:\t {}".format(model["common"]["image_size"]))
    log.log("Learning Rate:\t {}".format(model["opt"]["lr"]))
    log.log("Loss function:\t {}".format(model["opt"]["loss"]))
    if "weight" in locals():
        log.log("Weights :\t {}".format(dataset["weights"]["values"]))
    log.log("---")

    for epoch in range(resume, num_epochs):
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

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def get_dataset_loaders(model, dataset, workers):
    target_size = (model["common"]["image_size"],) * 2
    batch_size = model["common"]["batch_size"]
    path = dataset["common"]["dataset"]

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    train_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "training", "images")], os.path.join(path, "training", "labels"), transform
    )

    val_dataset = SlippyMapTilesConcatenation(
        [os.path.join(path, "validation", "images")], os.path.join(path, "validation", "labels"), transform
    )

    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader
