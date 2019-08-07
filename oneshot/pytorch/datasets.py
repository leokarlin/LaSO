import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ZippedDataLoader(DataLoader):
    """Wrapper class for zipping together several dataloaders.
    """
    def __init__(self, *data_loaders):
        self.data_loaders = data_loaders

    def __setattr__(self, attr, val):
        if attr == "data_loaders":
            super(ZippedDataLoader, self).__setattr__(attr, val)
        else:
            for data_loader in self.data_loaders:
                data_loader.__setattr__(attr, val)

    def __iter__(self):
        return zip(*[d.__iter__() for d in self.data_loaders])

    def __len__(self):
        return min(len(d) for d in self.data_loaders)


class InverseDataset(Dataset):
    """Wrapper class for inverting a dataset
    """

    def __init__(self, dataset : Dataset):

        self.dataset = dataset

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        output = self.dataset[len(self.dataset) - index - 1]

        return output

    def __len__(self) -> int:
        return len(self.dataset)


class FocalLoss(nn.Module):
    """Implement Focal Loss.

    Args:
        gamma (int, optional):
    """
    def __init__(self, gamma: int=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def focal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Focal loss

        Args:
          x: (tensor) sized [N, D].
          y: (tensor) sized [N, D].

        Return:
          (tensor) focal loss.
        """

        p = torch.sigmoid(x)
        pt = p * y + (1 - p) * (1 - y)
        w = (1 - pt).pow(self.gamma)
        w = F.normalize(w, p=1, dim=1)
        w.requires_grad = False

        return F.binary_cross_entropy_with_logits(x, y, w, reduction='sum')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between preds targets.

        Args:
          preds: (tensor) predicted labels, sized [batch_size, classes_num].
          targets: (tensor) target labels, sized [batch_size, classes_num].
        """

        cls_loss = self.focal_loss(preds, targets)

        return cls_loss