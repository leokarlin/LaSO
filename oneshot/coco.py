import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

from PIL import Image


COCO_LABELS_NUM = 80


def get_coco_labels(coco_ann: COCO) -> Tuple[dict, dict]:
    """Get the mapping between category and label.

    The coco categories ids are numbers between 1-90. This function maps
    these to labels between 0-79.
    """

    categories = coco_ann.loadCats(coco_ann.getCatIds())
    categories.sort(key=lambda x: x['id'])

    classes = {}
    id2label_map = {}

    for cat in categories:
        id2label_map[cat['id']] = len(classes)
        classes[len(classes)] = cat['name']

    return id2label_map, classes


def load_paths_multilabels(annotation_file: Path, imgs_base: Path) -> Tuple[list, list, dict]:
    """Load the paths to coco images and the multi-labels.

    """
    coco_ann = COCO(annotation_file=annotation_file)

    id2label_map, classes = get_coco_labels(coco_ann)

    imgs_paths, imgs_labels = [], []
    for img_id, img_data in coco_ann.imgs.items():
        img_anns_ids = coco_ann.getAnnIds(imgIds=[img_id])
        img_anns = coco_ann.loadAnns(img_anns_ids)

        #
        # Filter annotations with empty bounding box.
        #
        img_labels = {
            id2label_map[ann["category_id"]] for ann in img_anns \
            if ann["bbox"][2] > 0 and ann["bbox"][3] > 0
        }

        imgs_paths.append(str(imgs_base / img_data["file_name"]))
        imgs_labels.append(list(img_labels))

    return imgs_paths, imgs_labels, classes


class CocoMlDataset(Dataset):
    """Dataset class for the Multilabel COCO dataset
    """

    def __init__(self, image_paths: list, labels: list, transform: transforms.Compose = None,
                 categories_num: int = COCO_LABELS_NUM):

        self.image_paths = image_paths
        self.labels = labels
        self.categories_num = categories_num

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.array]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path = self.image_paths[index]
        target_inds = self.labels[index]
        target = np.zeros(self.categories_num, dtype=np.float32)
        target[target_inds] = 1

        sample = Image.open(image_path)
        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)


def copy_coco_data(force=False):
    """Copy the coco data to local machine for speed of reading.

    Args:
        force (bool, optional): Force copying coco data even if tmp folder exists.
    """

    if not os.path.exists("/tmp/aa"):
        os.system("mkdir /tmp/aa")

    if not os.path.exists("/tmp/aa/coco") or force:
        logging.info("Copying data to tmp")
        os.system("mkdir /tmp/aa/coco")
        os.system("mkdir /tmp/aa/coco/images")

        #
        # Copy the train data.
        #
        os.system("curl -o /tmp/aa/coco/train2014.zip FILE:/dccstor/faceid/data/oneshot/coco/train2014.zip")
        os.system(
            """unzip -o /tmp/aa/coco/train2014.zip -d /tmp/aa/coco/images | awk 'BEGIN {ORS=" "} {if(NR%100==0)print "."}'""")
        os.remove("/tmp/aa/coco/train2014.zip")

        #
        # Copy the train data.
        #
        os.system("curl -o /tmp/aa/coco/val2014.zip FILE:/dccstor/faceid/data/oneshot/coco/val2014.zip")
        os.system(
            """unzip -o /tmp/aa/coco/val2014.zip -d /tmp/aa/coco/images | awk 'BEGIN {ORS=" "} {if(NR%100==0)print "."}'""")
        os.remove("/tmp/aa/coco/val2014.zip")

        #
        # Copy the annotation data.
        #
        os.system(
            "curl -o /tmp/aa/coco/annotations_trainval2014.zip FILE:/dccstor/faceid/data/oneshot/coco/annotations_trainval2014.zip")
        os.system(
            """unzip -o /tmp/aa/coco/annotations_trainval2014.zip -d /tmp/aa/coco | awk 'BEGIN {ORS=" "} {if(NR%100==0)print "."}'""")
        os.remove("/tmp/aa/coco/annotations_trainval2014.zip")


