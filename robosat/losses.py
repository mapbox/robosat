"""PyTorch-compatible losses and loss functions.
"""

import torch
import torch.nn as nn

from torchvision.transforms.functional import normalize
from torchvision.models import vgg16_bn

from robosat.hooks import FeatureHook


class CrossEntropyLoss2d(nn.Module):
    """Cross-entropy.

    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    """Focal Loss.

    Reduces loss for well-classified samples putting focus on hard mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(penalty * nn.functional.log_softmax(inputs, dim=1), targets)


class mIoULoss2d(nn.Module):
    """mIoU Loss.

    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()

        softs = nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1).permute(1, 0, 2, 3)

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        miou = 1. - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()

        return max(miou, self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets))


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.

    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1)

        loss = 0.

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):

            max_margin_errors = 1. - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1. - labels_sorted).cumsum(0)
            iou = 1. - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N


class CombinedLoss(nn.Module):
    """Weighted combination of losses.
    """

    def __init__(self, criteria, weights):
        """Creates a `CombinedLosses` instance.

        Args:
          criteria: list of criteria to combine
          weights: tensor to tune losses with
        """

        super().__init__()

        assert len(weights.size()) == 1
        assert weights.size(0) == len(criteria)

        self.criteria = criteria
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0.0

        for criterion, w in zip(self.criteria, self.weights):
            each = w * criterion(inputs, targets)
            print(type(criterion).__name__, each.item())  # Todo: remove
            loss += each

        return loss


class TopologyLoss(nn.Module):
    """Topology loss working on a pre-trained model's feature map similarities.

    See:
      - https://arxiv.org/abs/1603.08155
      - https://arxiv.org/abs/1712.02190

    Note: implementation works with single channel tensors and stacks them for VGG.
    """

    def __init__(self, blocks, weights):
        """Creates a `TopologyLoss` instance.

        Args:
          blocks: list of block indices to use, in `[0, 6]` (e.g. `[0, 1, 2]`)
          weights: tensor to tune losses per block (e.g. `[0.2, 0.6, 0.2]`)

        Note: the block indices correspond to a pre-trained VGG's feature maps to use.
        """

        super().__init__()

        assert len(weights.size()) == 1
        assert weights.size(0) == len(blocks)

        self.weights = weights

        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        features = vgg16_bn(pretrained=True).features
        features.eval()

        for param in features.parameters():
            param.requires_grad = False

        relus = [i - 1 for i, m in enumerate(features) if isinstance(m, nn.MaxPool2d)]

        self.hooks = [FeatureHook(features[relus[i]]) for i in blocks]

        # Trim off unused layers to make forward pass more efficient
        self.features = features[0 : relus[blocks[-1]] + 1]

    def forward(self, inputs, targets):
        # model output to foreground probabilities
        inputs = nn.functional.softmax(inputs, dim=1)
        # we need to clone the tensor here before slicing otherwise pytorch
        # will lose track of information required for gradient computation
        inputs = inputs.clone()[:, 1, :, :]

        # masks are longs but vgg wants floats
        targets = targets.float()

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = normalize(inputs, mean, std)
        targets = normalize(targets, mean, std)

        # N, H, W -> N, C, H, W
        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(1)

        # repeat channel three times for using a pre-trained three-channel VGG
        inputs = inputs.repeat(1, 3, 1, 1)
        targets = targets.repeat(1, 3, 1, 1)

        # extract feature maps and compare their weighted loss

        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += nn.functional.mse_loss(lhs, rhs) * w

        return loss

    def close(self):
        for hook in self.hooks:
            hook.close()
