import os
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from robosat.utils.core import seed_rngs
from robosat.utils.config import load_config
from robosat.model.datasets import SlippyMapTiles
from robosat.model.samplers import RandomSubsetSampler
from robosat.model.transforms import ConvertImageMode, MaskToTensor


def add_parser(subparser):
    parser = subparser.add_parser('weights', help='computes class weights on dataset',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, required=True, help='path to dataset configuration file')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')

    parser.set_defaults(func=main)


def main(args):
    seed_rngs(args.seed)

    dataset = load_config(args.dataset)

    path = dataset['common']['dataset']
    num_classes = len(dataset['common']['classes'])

    train_transform = Compose([
        ConvertImageMode(mode='P'),
        MaskToTensor()
    ])

    train_dataset = SlippyMapTiles(os.path.join(path, 'training', 'labels'), transform=train_transform)
    train_sampler = RandomSubsetSampler(train_dataset, dataset['samples']['training'])

    n = 0
    counts = np.zeros(num_classes, dtype=np.int64)

    loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
    for images, tile in tqdm(loader, desc='Loading', unit='image', ascii=True):
        image = torch.squeeze(images)

        image = np.array(image, dtype=np.uint8)
        n += image.shape[0] * image.shape[1]
        counts += np.bincount(image.ravel(), minlength=num_classes)

    # Class weighting scheme `w = 1 / ln(c + p)` see:
    # - https://arxiv.org/abs/1707.03718
    #     LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    # - https://arxiv.org/abs/1606.02147
    #     ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation

    probs = counts / n
    weights = 1 / np.log(1.02 + probs)

    weights.round(6, out=weights)
    print(weights.tolist())
