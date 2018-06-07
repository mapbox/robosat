import random

import torch.utils.data.sampler


class RandomSubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, samples):
        n = len(dataset)
        self.indices = random.sample(range(n), k=min(samples, n))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
