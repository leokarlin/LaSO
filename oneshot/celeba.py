from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


CELEBA_LABELS_NUM = 40
CELEBA_IDENTITIES_NUM = 10177


def load_paths_multilabels(
        annotation_file: Union[Path, str],
        partition_file: Union[Path, str],
        imgs_base: Union[Path, str]) -> Tuple[list, np.array, list, np.array]:
    """
    Load the paths and multi-labels of the Celeb-A dataset.

    Args:
        annotation_file:
        partition_file:
        imgs_base:

    Returns:
        train_imgs_paths, train_imgs_labels, val_imgs_paths, val_imgs_labels
    """
    att_table = pd.read_csv(
        annotation_file,
        delim_whitespace=True,
        skiprows=1
    )
    att_table[att_table == -1] = 0

    part_tbl = pd.read_csv(partition_file, delim_whitespace=True, names=["partition"])

    train_imgs_paths = [imgs_base / s for s in att_table[part_tbl["partition"]==0].index]
    val_imgs_paths = [imgs_base / s for s in att_table[part_tbl["partition"]==1].index]
    train_imgs_labels = att_table[part_tbl["partition"]==0].values.astype(np.uint8)
    val_imgs_labels = att_table[part_tbl["partition"]==1].values.astype(np.uint8)

    return train_imgs_paths, train_imgs_labels, val_imgs_paths, val_imgs_labels


def load_paths_ids_multilabels(
        annotation_file: Union[Path, str],
        partition_file: Union[Path, str],
        identity_file: Union[Path, str],
        imgs_base: Union[Path, str]) -> Tuple[list, np.array, np.array, list, np.array, np.array, list, np.array, np.array]:
    """
    Load the paths and multi-labels of the Celeb-A dataset.

    Args:
        annotation_file:
        partition_file:
        imgs_base:

    Returns:
        train_imgs_paths, train_imgs_labels, val_imgs_paths, val_imgs_labels
    """
    att_table = pd.read_csv(
        annotation_file,
        delim_whitespace=True,
        skiprows=1
    )
    att_table[att_table == -1] = 0

    identity_tbl = pd.read_csv(identity_file, delim_whitespace=True, names=["identity"])
    identity_tbl["identity"] -= 1

    part_tbl = pd.read_csv(partition_file, delim_whitespace=True, names=["partition"])

    train_imgs_paths = [imgs_base / s for s in att_table[part_tbl["partition"]==0].index]
    train_imgs_identities = identity_tbl[part_tbl["partition"]==0].values.astype(np.int64)
    train_imgs_labels = att_table[part_tbl["partition"]==0].values.astype(np.uint8)

    val_imgs_paths = [imgs_base / s for s in att_table[part_tbl["partition"]==1].index]
    val_imgs_identities = identity_tbl[part_tbl["partition"]==1].values.astype(np.int64)
    val_imgs_labels = att_table[part_tbl["partition"]==1].values.astype(np.uint8)

    test_imgs_paths = [imgs_base / s for s in att_table[part_tbl["partition"]==2].index]
    test_imgs_identities = identity_tbl[part_tbl["partition"]==2].values.astype(np.int64)
    test_imgs_labels = att_table[part_tbl["partition"]==2].values.astype(np.uint8)

    return train_imgs_paths, train_imgs_identities, train_imgs_labels, val_imgs_paths, val_imgs_identities, \
           val_imgs_labels, test_imgs_paths, test_imgs_identities, test_imgs_labels


class CelebIdsDataset(Dataset):
    """
    Dataset class for the Multilabel Celeb-A dataset.

    Args:
        image_paths:
        labels:
        transform:
        categories_num:
    """

    def __init__(self, image_paths: list, identities: np.array, transform: transforms.Compose = None):
        self.image_paths = image_paths
        self.identities = identities
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target).
        """
        image_path = self.image_paths[index]
        target = self.identities[index]

        sample = Image.open(image_path)
        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)



class CelebMlDataset(Dataset):
    """
    Dataset class for the Multilabel Celeb-A dataset.

    Args:
        image_paths:
        labels:
        transform:
        categories_num:
    """

    def __init__(self, image_paths: list, labels: np.array, transform: transforms.Compose = None,
                 categories_num: int = CELEBA_LABELS_NUM):
        self.image_paths = image_paths
        self.labels = labels
        self.categories_num = categories_num

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target).
        """
        image_path = self.image_paths[index]
        target = self.labels[index].astype(np.float32)

        sample = Image.open(image_path)
        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)


class CelebMlIdsDataset(Dataset):
    """
    Dataset class for the Multilabel Celeb-A dataset.

    Args:
        image_paths:
        labels:
        transform:
        categories_num:
    """

    def __init__(self, image_paths: list, identities: np.array, labels: np.array, transform: transforms.Compose = None,
                 categories_num: int = CELEBA_LABELS_NUM):
        self.image_paths = image_paths
        self.identities = identities
        self.labels = labels
        self.categories_num = categories_num

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target).
        """
        image_path = self.image_paths[index]
        identity = self.identities[index]
        attr = self.labels[index].astype(np.float32)

        sample = Image.open(image_path)
        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, identity, attr

    def __len__(self) -> int:
        return len(self.image_paths)


class CelebAccDataset(Dataset):
    """
    Dataset class for calculating the accuracy Celeb-A dataset.

    Args:
        image_paths0:
        image_paths1:
        labels:
        transform:
    """

    def __init__(
            self,
            image_paths0: list,
            image_paths1: list,
            image_attrs0: np.array,
            image_attrs1: np.array,
            labels: np.array,
            transform: transforms.Compose=None):

        self.image_paths0 = image_paths0
        self.image_paths1 = image_paths1
        self.image_attrs0 = image_attrs0.astype(np.float32)
        self.image_attrs1 = image_attrs1.astype(np.float32)
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image, int, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, image, target).
        """
        image_path0 = self.image_paths0[index]
        image_path1 = self.image_paths1[index]

        image_attrs0 = self.image_attrs0[index]
        image_attrs1 = self.image_attrs1[index]

        target = self.labels[index]

        sample0 = Image.open(image_path0).convert(mode='RGB')
        sample1 = Image.open(image_path1).convert(mode='RGB')

        if self.transform is not None:
            sample0 = self.transform(sample0)
            sample1 = self.transform(sample1)

        return sample0, sample1, target, image_attrs0, image_attrs1

    def __len__(self) -> int:
        return len(self.image_paths0)


class CelebEmbeddingDataset(Dataset):
    """
    Dataset class running on the Celeb-A dataset.

    Args:
        data_path (list): Path to the embedding data.
        return_ids (bool): Whether to return the ids of the emebeddings.
        return_attr (bool): Whether to return the attributes of the emebeddings.
    """

    def __init__(self, data_paths: List[str], return_ids: bool=True, return_attr: bool=True):

        self.data_index = 0
        self.data_paths = data_paths
        self.return_ids = return_ids
        self.return_attr = return_attr
        self.load_embeddings()

    def load_embeddings(self):

        data_path = self.data_paths[self.data_index]
        self.data_index = (self.data_index + 1) % len(self.data_paths)

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.embeddings: np.array = data["embeddings"]
        self.identities: np.array = data["identities"]
        self.targets: np.array = data["targets"]

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple of embedding, identity, target.
        """

        return_value = [self.embeddings[index]]

        if self.return_ids:
            return_value += [self.identities[index]]

        if self.return_attr:
            return_value += [self.targets[index]]

        if len(return_value) == 1:
            return_value  = return_value[0]

        return return_value

    def __len__(self) -> int:
        return len(self.embeddings)


class LFWDataset(Dataset):
    """Dataset class for testing and embedding on the LFW dataset.

    Note:
        Pairs of similar identity have target 1. Distinct identities have target 0.

    Args:
        image_paths:
        labels:
        transform:
    """

    def __init__(
            self,
            base_path: Path=Path("/u/amitaid/scikit_learn_data/lfw_home/lfw_funneled/"),
            pairs_table_path: Path=Path("/u/amitaid/scikit_learn_data/lfw_home/pairs.txt"),
            transform: transforms.Compose = None):

        data = []
        with pairs_table_path.open() as f:
            _ = f.readline()
            for i in range(10):
                for j in range(600):
                    if j < 300:
                        name0, id0, id1 = f.readline().strip().split("\t")
                        name1 = name0
                        target = 1
                    else:
                        name0, id0, name1, id1 = f.readline().strip().split("\t")
                        target = 0

                    data.append(
                        (
                            "{}/{}_{:04}.jpg".format(name0, name0, int(id0)),
                            "{}/{}_{:04}.jpg".format(name1, name1, int(id1)),
                            target
                        )
                    )

        self.data = data
        self.base_path = base_path
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target).
        """

        image_path_a, image_path_b, target = self.data[index]
        target = np.array(target).astype(np.float32)

        image_a = Image.open(self.base_path / image_path_a)
        if image_a.mode != 'RGB':
            image_a = image_a.convert(mode='RGB')

        image_b = Image.open(self.base_path / image_path_b)
        if image_b.mode != 'RGB':
            image_b = image_b.convert(mode='RGB')

        if self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        return image_a, image_b, target

    def __len__(self) -> int:
        return len(self.data)