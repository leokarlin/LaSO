"""
Pytorch datasets for loading COCO.
"""
from collections import defaultdict
import itertools
import logging
import os
import pickle
import random

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import trange


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

COCO_CLASS_NUM = 80


def labels_list_to_1hot(labels_list, class_list):
    """Convert a list of indices to 1hot representation."""

    labels_1hot = np.zeros(COCO_CLASS_NUM, dtype=np.float32)

    labels_1hot[list(filter(lambda x: x in class_list, labels_list))] = 1
    assert labels_1hot.sum() > 0, "No labels in conversion of labels list to 1hot"

    return labels_1hot


def load_image(img_path):
    img = Image.open(img_path)

    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')

    return img


class CocoDataset(Dataset):
    def __init__(
            self,
            root_dir,
            set_name='train2014',
            unseen_set=False,
            transform=None,
            return_ids=False,
            debug_size=-1):
        """COCO Dataset

        Args:
            root_dir (string): COCO directory.
            set_name (string): 'train2014'/'val2014'.
            unseen_set (bool): Whether to use the seen (64 classes) or unseen (16) classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            return_ids (bool, optional): Whether to return also the image ids.
            debug_size (int, optional): Subsample the dataset. Useful for debug.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.unseen_set = unseen_set
        self.return_ids = return_ids
        self.deubg_size = debug_size

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.setup_classes()
        self.load_classes()

        self.calc_indices()

    def setup_classes(self):
        """Setup the seen/unseen classes and labels/images mappings."""

        #
        # Create a random (up to seed) set of (64/16 out of 80) classes.
        #
        random.seed(0)
        self.labels_list = set(random.sample(range(80), 64))
        if self.unseen_set:
            self.labels_list = set(range(80)) - self.labels_list
        self.labels_list = sorted(list(self.labels_list))

    def load_classes(self):
        """Load class/categories/labels and create mapping."""

        #
        # Load class names (name -> label)
        #
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.label_to_category = {i: c['id'] for i, c in enumerate(categories)}
        self.category_to_label = {v: k for k, v in self.label_to_category.items()}
        self.label_to_class = {i: c['name'] for i, c in enumerate(categories)}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}

    def calc_indices(self):
        """Setup the filtered images lists."""

        #
        # Map classes/labels to image indices.
        #
        labels_to_img_ids = defaultdict(list)
        for image_id in self.image_ids:
            labels = set(self.load_labels(image_id))

            #
            # Filter images with no labels.
            #
            if not labels:
                continue

            #
            # Map labels to their images (ids).
            #
            for label in labels:
                labels_to_img_ids[label].append(image_id)

        #
        # Filter images according to labels list.
        #
        image_ids_lists = [labels_to_img_ids[l] for l in self.labels_list]
        self.image_ids = list(set(itertools.chain(*image_ids_lists)))

        #
        # Create the image paths and labels list (for the dataset __getitem__ method).
        #
        self.image_paths = []
        self.image_labels = []
        for image_id in self.image_ids:
            self.image_paths.append(self.image_id_to_path(image_id))
            self.image_labels.append(
                labels_list_to_1hot(self.load_labels(image_id), self.labels_list)
            )

    def load_labels(self, image_id):
        """Get the labels of an image by image_id."""

        #
        # get ground truth annotations
        #
        annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)

        #
        # some images appear to miss annotations (like image with id 257034)
        #
        if len(annotations_ids) == 0:
            return []

        #
        # parse annotations
        #
        coco_annotations = self.coco.loadAnns(annotations_ids)
        labels = [self.category_to_label[ca['category_id']] \
                       for ca in coco_annotations if ca['bbox'][2] > 0 and ca['bbox'][3] > 0]

        return sorted(set(labels))

    def image_id_to_path(self, img_id):
        """Convert image ids to paths."""

        #
        # Note:
        # The reason I use `int` is that `img_id` might be numpy scalar, and coco
        # doesn't like it.
        #
        image_info = self.coco.loadImgs(int(img_id))[0]
        return os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])

    def __len__(self):

        if self.deubg_size > 0:
            return self.deubg_size

        return len(self.image_ids)

    def __getitem__(self, idx):

        img = load_image(self.image_paths[idx])
        labels = self.image_labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.return_ids:
            return img, labels, self.image_ids[idx]

        return img, labels


class CocoDatasetPairs(CocoDataset):
    """Create pairs of images from the Coco dataset.

    The image pairs are calculated so as to have at least on label intersection.

    Args:
        root_dir (string): COCO directory.
        set_name (string): 'train2014'/'val2014'.
        unseen_set (bool): Whether to use the seen (64 classes) or unseen (16) classes.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        return_ids (bool, optional): Whether to return also the image ids.
        debug_size (int, optional): Subsample the dataset. Useful for debug.
        dataset_size_ratio (int, optional): Multiplier of dataset size.

    Note:
        Contrary to the `CocoDataset`, this dataset might include images with labels
        other than the seen (or unseen). But the labels themselves are not outputed.
        This means that the trained networks have seen the images, but don't know
        anything about the labels.
    """

    def __init__(
            self,
            root_dir,
            set_name='train2014',
            unseen_set=False,
            transform=None,
            return_ids=False,
            debug_size=-1,
            dataset_size_ratio=1):

        self.dataset_size_ratio = dataset_size_ratio

        super(CocoDatasetPairs, self).__init__(
            root_dir=root_dir,
            set_name=set_name,
            unseen_set=unseen_set,
            transform=transform,
            return_ids=return_ids,
            debug_size=debug_size
        )

    def calc_indices(self):
        """Pre-calculate all pair indices.

        Note:
            Should be called at the start of each epoch.
        """

        #
        # Map classes/labels to image indices.
        #
        self.labels_to_img_ids = defaultdict(list)
        for image_id in self.image_ids:
            labels = self.load_labels(image_id)
            if not labels:
                continue

            for label in labels:
                self.labels_to_img_ids[label].append(image_id)

        self.class_histogram = np.zeros(COCO_CLASS_NUM)

        #
        # Pre-process the list of training samples (A, B images for set-operations).
        # Note:
        # The length of the pairs list is set arbitrarily to the number of the
        # images in the original dataset.
        #
        logging.info("Calculating indices.")
        self.images_indices = []
        for i in trange(self.dataset_size_ratio * len(self.image_ids)):
            self.images_indices.append(self.calc_index(i))

    def calc_index(self, idx):
        """Calculate a pair of samples for training."""

        while True:
            #
            # The first index is select randomly among the labels with the minimum
            # samples so far.
            #
            min_label = np.random.choice(self.labels_list)
            for ind in range(COCO_CLASS_NUM):
                if ind in self.labels_list:
                    if self.class_histogram[ind] < self.class_histogram[min_label]:
                        min_label = ind

            tmp_idx = idx % len(self.labels_to_img_ids[min_label])
            img_id1 = self.labels_to_img_ids[min_label][tmp_idx]
            labels1 = self.load_labels(img_id1)

            if labels1:
                break

            idx = np.random.randint(len(self.image_ids))

        labels1 = labels_list_to_1hot(labels1, self.labels_list)

        one_indices = np.where(labels1 == 1)[0]
        self.class_histogram[one_indices] += 1

        #
        # The second index is selected from the images that share labels
        # with the first index, but has minimal sampling so far.
        #
        min_label = one_indices[0]
        for ind in one_indices:
            if self.class_histogram[ind] < self.class_histogram[min_label]:
                min_label = ind

        img_id2 = np.random.choice(self.labels_to_img_ids[min_label])

        labels2 = self.load_labels(img_id2)
        labels2 = labels_list_to_1hot(labels2, self.labels_list)

        one_indices = np.where(labels2 == 1)[0]
        self.class_histogram[one_indices] += 1

        return self.image_id_to_path(img_id1), self.image_id_to_path(img_id2), labels1, labels2, img_id1, img_id2

    def __len__(self):

        if self.deubg_size > 0:
            return self.deubg_size

        return len(self.images_indices)

    def __getitem__(self, idx):

        path1, path2, labels1, labels2, id1, id2 = self.images_indices[idx]

        img1 = load_image(path1)
        img2 = load_image(path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.return_ids:
            return img1, img2, labels1, labels2, id1, id2

        return img1, img2, labels1, labels2


class CocoDatasetPairsSub(CocoDatasetPairs):
    """Create pairs of images from the Coco dataset.

    The image pairs are calculated so as to have at least on label intersection.

    Args:
        root_dir (string): COCO directory.
        set_name (string): 'train2014'/'val2014'.
        unseen_set (bool): Whether to use the seen (64 classes) or unseen (16) classes.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        return_ids (bool, optional): Whether to return also the image ids.
        debug_size (int, optional): Subsample the dataset. Useful for debug.
        dataset_size_ratio (int, optional): Multiplier of dataset size.

    Note:
        Contrary to the `CocoDataset`, this dataset might include images with labels
        other than the seen (or unseen). But the labels themselves are not outputed.
        This means that the trained networks have seen the images, but don't know
        anything about the labels.
    """

    def __init__(
            self,
            root_dir,
            set_name='train2014',
            unseen_set=False,
            transform=None,
            return_ids=False,
            debug_size=-1,
            dataset_size_ratio=1):

        self.dataset_size_ratio = dataset_size_ratio

        super(CocoDatasetPairsSub, self).__init__(
            root_dir=root_dir,
            set_name=set_name,
            unseen_set=unseen_set,
            transform=transform,
            return_ids=return_ids,
            debug_size=debug_size
        )

    def calc_index(self, idx):
        from oneshot.alfassy import get_subtraction_exp
        """Calculate a pair of samples for training."""
        while True:
            while True:
                #
                # The first index is select randomly among the labels with the minimum
                # samples so far.
                #
                min_label = np.random.randint(COCO_CLASS_NUM)
                for ind in range(COCO_CLASS_NUM):
                    if ind in self.labels_list:
                        if self.class_histogram[ind] < self.class_histogram[min_label]:
                            min_label = ind

                tmp_idx = idx % len(self.labels_to_img_ids[min_label])
                img_id1 = self.labels_to_img_ids[min_label][tmp_idx]
                labels1exp = self.load_labels(img_id1)
                labels1exp = list(set(labels1exp))
                filteredLabels1 = [label for label in labels1exp if label in self.labels_list]

                if filteredLabels1:
                    break
                idx = np.random.randint(len(self.image_ids))

            labels1 = labels_list_to_1hot(labels1exp, self.labels_list)

            one_indices = np.where(labels1 == 1)[0]
            # self.class_histogram[one_indices] += 1

            #
            # The second index is selected from the images that share labels
            # with the first index, but has minimal sampling so far.
            #
            min_label = one_indices[0]
            for ind in one_indices:
                if self.class_histogram[ind] < self.class_histogram[min_label]:
                    min_label = ind

            # img_id2 = np.random.choice(self.labels_to_img_ids[min_label])
            tmp_idx2 = np.random.randint(len(self.labels_to_img_ids[min_label]))
            img_id2 = self.labels_to_img_ids[min_label][tmp_idx2]

            labels2exp = self.load_labels(img_id2)
            labels2exp = list(set(labels2exp))
            filteredLabels2 = [label for label in labels2exp if label in self.labels_list]

            labels2 = labels_list_to_1hot(labels2exp, self.labels_list)

            sub_labels = get_subtraction_exp(filteredLabels1, filteredLabels2)
            continue_test = len(sub_labels)
            if continue_test == 0:
                idx = np.random.randint(len(self.image_ids))
            else:
                one_indices = np.where(labels1 == 1)[0]
                self.class_histogram[one_indices] += 1
                one_indices = np.where(labels2 == 1)[0]
                self.class_histogram[one_indices] += 1
                break

        return self.image_id_to_path(img_id1), self.image_id_to_path(img_id2), labels1, labels2, img_id1, img_id2


class CocoDatasetTriplets(CocoDatasetPairs):
    """Create "Triplets" of images from the Coco dataset.

    First image pairs are calculated so as to have at least on label intersection. Then
    for each setop operation I, U, S, S (reversed) a triplet image is fonud that has
    the correct labels that the setop produces.

    Args:
        root_dir (string): COCO directory.
        set_name (string): 'train2014'/'val2014'.
        unseen_set (bool): Whether to use the seen (64 classes) or unseen (16) classes.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        return_ids (bool, optional): Whether to return also the image ids.
        debug_size (int, optional): Subsample the dataset. Useful for debug.

    Note:
        Contrary to the `CocoDataset`, this dataset might include images with labels
        other than the seen (or unseen). But the labels themselves are not outputed.
        This means that the trained networks have seen the images, but don't know
        anything about the labels.
    """

    def calc_indices(self):
        """Pre-calculate all pair indices.

        Note:
            Should be called at the start of each epoch
        """

        #
        # Map classes/labels to image indices.
        #
        self.labels_to_img_ids = defaultdict(list)
        self.labels_num_to_img_ids = defaultdict(list)

        for image_id in self.image_ids:
            labels = self.load_labels(image_id)
            if not labels:
                continue

            self.labels_num_to_img_ids[len(labels)].append(image_id)

            for label in labels:
                self.labels_to_img_ids[label].append(image_id)

        self.labels_to_img_ids_sets = {k: set(v) for k, v in self.labels_to_img_ids.items()}
        self.labels_num_to_img_ids = {k: set(v) for k, v in self.labels_num_to_img_ids.items()}

        self.class_histogram = np.zeros(COCO_CLASS_NUM)

        #
        # Pre-process the list of training samples (A, B images for set-operations).
        # Note:
        # The length of the pairs list is set arbitrarily to the number of the
        # images in the original dataset.
        #
        logging.info("Calculating indices.")
        self.images_indices = []
        for i in trange(len(self.image_ids)):
            self.images_indices.append(self.calc_index(i))

    def calc_index(self, idx):
        """Calculate a pair of samples for training."""

        while True:
            #
            # The first index is select randomly among the labels with the minimum
            # samples so far.
            #
            min_label = np.random.choice(self.labels_list)
            for ind in range(COCO_CLASS_NUM):
                if ind in self.labels_list:
                    if self.class_histogram[ind] < self.class_histogram[min_label]:
                        min_label = ind

            tmp_idx = idx % len(self.labels_to_img_ids[min_label])
            img_id1 = self.labels_to_img_ids[min_label][tmp_idx]
            labels1 = self.load_labels(img_id1)

            if labels1:
                break

            idx = np.random.randint(len(self.image_ids))

        labels1 = labels_list_to_1hot(labels1, self.labels_list)

        one_indices = np.where(labels1 == 1)[0]
        self.class_histogram[one_indices] += 1

        #
        # The second index is selected from the images that share labels
        # with the first index, but has minimal sampling so far.
        #
        min_label = one_indices[0]
        for ind in one_indices:
            if self.class_histogram[ind] < self.class_histogram[min_label]:
                min_label = ind

        img_id2 = np.random.choice(self.labels_to_img_ids[min_label])

        labels2 = self.load_labels(img_id2)
        labels2 = labels_list_to_1hot(labels2, self.labels_list)

        one_indices = np.where(labels2 == 1)[0]
        self.class_histogram[one_indices] += 1

        return_values = [self.image_id_to_path(img_id1), self.image_id_to_path(img_id2), labels1, labels2]

        #
        # Find setop triplet examples for each setop operations (union, intersection, 2x substraction)
        #
        b_lbls1, b_lbls2 = labels1.astype(np.bool), labels2.astype(np.bool)
        for op_lbls in ((b_lbls1 & b_lbls2), (b_lbls1 | b_lbls2), (b_lbls1 & ~b_lbls2), (b_lbls2 & ~b_lbls1)):
            op_lbls = np.where(op_lbls)[0]
            if len(op_lbls) > 0:
                img_id = self.find_imgs_by_labels(op_lbls)
                path = self.image_id_to_path(img_id) if img_id else None
            else:
                path = None
            return_values.extend([path, np.float32(1 if path else 0)])

        return return_values

    def find_imgs_by_labels(self, labels):
        """Find images that fit a set of labels."""

        #
        # Find images with the set of labels.
        #
        img_set = self.labels_to_img_ids_sets[labels[0]]
        for label in labels[1:]:
            img_set = img_set.intersection(self.labels_to_img_ids_sets[label])

        #
        # Verify that they contain only the set of required labels.
        # This is done by verifying the number of labels (might be a more efficient way).
        #
        if len(labels) in self.labels_num_to_img_ids:
            img_set = img_set.intersection(self.labels_num_to_img_ids[len(labels)])
        else:
            img_set = None

        if img_set:
            img_id = random.choice(list(img_set))
        else:
            img_id = None

        return img_id

    def __getitem__(self, idx):

        path1, path2, labels1, labels2, path3, weight3, path4, weight4, \
        path5, weight5, path6, weight6 = self.images_indices[idx]

        transform = self.transform if self.transform else lambda x: x

        imgs = [transform(load_image(path)) if path else torch.zeros(3, 299, 299) for path in
                [path1, path2, path3, path4, path5, path6]]

        return imgs[0], imgs[1], labels1, labels2, imgs[2], weight3, imgs[3], \
               weight4, imgs[4], weight5, imgs[5], weight6


class CocoDatasetAugment(CocoDataset):
    """Create samples for augmentation train/tests."""

    def __init__(
            self,
            root_dir,
            used_inds_path,
            set_name='train2014',
            transform=None,
            return_ids=False,
            debug_size=-1,
            ):

        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.unseen_set = True
        self.return_ids = return_ids
        self.deubg_size = debug_size

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        with open(used_inds_path, 'rb') as f:
            self.used_image_idxs = pickle.load(f)

        self.setup_classes()
        self.load_classes()

        self.calc_indices()

    def calc_indices(self):
        """Pre-calculate all indices.

        Note:
            Should be called at the start of each epoch.
        """

        self.fake_pairs = list(itertools.combinations(self.used_image_idxs, 2))

        #
        # Map classes/labels to image indices.
        #
        self.labels_to_img_ids = defaultdict(list)
        for image_idx in self.used_image_idxs:
            image_id = self.image_ids[image_idx]
            labels = self.load_labels(image_id)
            if not labels:
                continue

            for label in labels:
                self.labels_to_img_ids[label].append(image_id)

        self.class_histogram = np.zeros(COCO_CLASS_NUM)

        #
        # Pre-process the list of training samples (A, B images for set-operations).
        # Note:
        # The length of the pairs list is set arbitrarily to the number of the
        # images in the original dataset.
        #
        logging.info("Calculating indices.")
        self.images_indices = []
        for i in trange(len(self)):
            self.images_indices.append(self.calc_index(i))

    def calc_index(self, idx):
        """Calculate samples for training."""

        fake_idx1, fake_idx2 = self.fake_pairs[idx % len(self.fake_pairs)]
        fake_id1, fake_id2 = self.image_ids[fake_idx1], self.image_ids[fake_idx2]
        fake_labels1 = self.load_labels(fake_id1)
        fake_labels2 = self.load_labels(fake_id2)
        fake_labels1 = labels_list_to_1hot(fake_labels1, self.labels_list)
        fake_labels2 = labels_list_to_1hot(fake_labels2, self.labels_list)

        while True:
            #
            # The index is select randomly among the labels with the minimum
            # samples so far.
            #
            min_label = np.random.choice(self.labels_list)
            for ind in range(COCO_CLASS_NUM):
                if ind in self.labels_list:
                    if self.class_histogram[ind] < self.class_histogram[min_label]:
                        min_label = ind

            #
            # Select an image with the selected label.
            #
            tmp_idx = idx % len(self.labels_to_img_ids[min_label])
            img_id1 = self.labels_to_img_ids[min_label][tmp_idx]
            labels1 = self.load_labels(img_id1)

            #
            # Note:
            # I am not sure that there is a need to check that there is a need to check for a label here.
            #
            if labels1:
                break

            idx = np.random.randint(len(self))

        labels1 = labels_list_to_1hot(labels1, self.labels_list)

        one_indices = np.where(labels1 == 1)[0]
        self.class_histogram[one_indices] += 1

        return self.image_id_to_path(img_id1), labels1, img_id1, \
               self.image_id_to_path(fake_id1), fake_labels1, fake_id1, \
               self.image_id_to_path(fake_id2), fake_labels2, fake_id2

    def __getitem__(self, idx):

        path1, labels1, id1, path2, labels2, id2, path3, labels3, id3 = self.images_indices[idx]

        img1 = load_image(path1)
        img2 = load_image(path2)
        img3 = load_image(path3)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        if self.return_ids:
            return img1, labels1, id1, img2, labels2, id2, img3, labels3, id3

        return img1, labels1, img2, labels2, img3, labels3

    def __len__(self):

        if self.deubg_size > 0:
            return self.deubg_size

        return 4 * len(self.fake_pairs)
