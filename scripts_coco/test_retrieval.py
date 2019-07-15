"""Calculate retrieval on the seen classes of COCO."""
import logging
from more_itertools import chunked
import numpy as np


from pathlib import Path
import pickle
from tqdm import tqdm

from joblib import Parallel, delayed

import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from traitlets import Bool, Enum, Float, Int, Unicode

from oneshot import setops_models
from oneshot.setops_models import Inception3
from oneshot import alfassy
from oneshot.coco import copy_coco_data

from experiment import Experiment

from CCC import setupCUDAdevice

from ignite._utils import convert_tensor


setupCUDAdevice()

cuda = True if torch.cuda.is_available() else False
device = 'cuda'

#
# Seed the random states
#
np.random.seed(0)
random_state = np.random.RandomState(0)


def _prepare_batch(batch, device=None):
    return [convert_tensor(x, device=device) for x in batch]


def calc_IOU(y, y_pred):
    """Calculate Intersection Over Union between two multi labels vectors."""

    y = y.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    support = (y + y_pred) > 0.5
    correct = np.equal(y_pred, y)[support]

    return correct.sum() / (support.sum() + 1e-6)


def label2hash(label):
    hash = "".join([chr(i) for i in np.where(label==1)[0]])
    return hash


class Main(Experiment):

    description = Unicode(u"Calculate retrieval of trained coco model.")

    #
    # Run setup
    #
    batch_size = Int(256, config=True, help="Batch size. default: 256")
    num_workers = Int(8, config=True, help="Number of workers to use for data loading. default: 8")
    n_jobs = Int(-1, config=True, help="Number of workers to use for data loading. default: -1")
    device = Unicode("cuda", config=True, help="Use `cuda` backend. default: cuda")

    #
    # Hyper parameters.
    #
    unseen = Bool(False, config=True, help="Test on unseen classes.")
    skip_tests = Int(1, config=True, help="How many test pairs to skip? for better runtime. default: 1")
    debug_size = Int(-1, config=True, help="Reduce dataset sizes. This is useful when developing the script. default -1")

    #
    # Resume previous run parameters.
    #
    resume_path = Unicode(u"/dccstor/alfassy/finalLaSO/code_release/paperModels", config=True,
                          help="Resume from checkpoint file (requires using also '--resume_epoch'.")
    resume_epoch = Int(0, config=True, help="Epoch to resume (requires using also '--resume_path'.")
    coco_path = Unicode(u"/tmp/aa/coco", config=True, help="path to local coco dataset path")
    init_inception = Bool(True, config=True, help="Initialize the inception networks using paper's network.")

    #
    # Network hyper parameters
    #
    base_network_name = Unicode("Inception3", config=True, help="Name of base network to use.")
    avgpool_kernel = Int(10, config=True,
                         help="Size of the last avgpool layer in the Resnet. Should match the cropsize.")
    classifier_name = Unicode("Inception3Classifier", config=True, help="Name of classifier to use.")
    sets_network_name = Unicode("SetOpsResModule", config=True, help="Name of setops module to use.")
    sets_block_name = Unicode("SetopResBlock_v1", config=True, help="Name of setops network to use.")
    sets_basic_block_name = Unicode("SetopResBasicBlock", config=True,
                                    help="Name of the basic setops block to use (where applicable).")
    ops_layer_num = Int(1, config=True, help="Ops Module layer num.")
    ops_latent_dim = Int(1024, config=True, help="Ops Module inner latent dim.")
    setops_dropout = Float(0, config=True, help="Dropout ratio of setops module.")
    crop_size = Int(299, config=True, help="Size of input crop (Resnet 224, inception 299).")
    scale_size = Int(350, config=True, help="Size of input scale for data augmentation. default: 350")
    paper_reproduce = Bool(False, config=True, help="Use paper reproduction settings. default: False")

    #
    # Metric
    #
    tree_type = Enum(("BallTree", "KDTree"), config=True, default_value="BallTree",
                  help="The Nearest-Neighbour algorithm to use.  Default='BallTree'.")
    metric = Enum(("manhattan", "minkowski"), config=True, default_value="minkowski",
                  help="The distance metric to use for the BallTree.  Default='minkowski'.")

    def run(self):
        #
        # Setup the model
        #
        base_model, classifier, setops_model = self.setup_model()

        base_model.to(self.device)
        classifier.to(self.device)
        setops_model.to(self.device)

        base_model.eval()
        classifier.eval()
        setops_model.eval()

        #
        # Load the dataset
        #
        val_loader, pair_loader, pair_loader_sub = self.setup_datasets()

        val_labels, val_outputs = self.embed_dataset(base_model, val_loader)

        self.val_labels_set = set([label2hash(label) for label in val_labels])

        logging.info("Calculate the embedding NN {}.".format(self.tree_type))
        if self.tree_type == "BallTree":
            tree = BallTree(val_outputs, metric=self.metric)
        else:
            tree = KDTree(val_outputs)

        #
        # Run the testing
        #
        logging.info("Calculate test set embedding.")
        a_S_b_list, b_S_a_list, a_U_b_list, b_U_a_list, a_I_b_list, b_I_a_list = [], [], [], [], [], []
        target_a_I_b_list, target_a_U_b_list, target_a_S_b_list, target_b_S_a_list = [], [], [], []
        embed_a_list, embed_b_list, target_a_list, target_b_list = [], [], [], []
        ids_a_list, ids_b_list = [], []
        with torch.no_grad():
            for batch in tqdm(pair_loader):
                input_a, input_b, target_a, target_b, id_a, id_b = _prepare_batch(batch, device=self.device)

                ids_a_list.append(id_a.cpu().numpy())
                ids_b_list.append(id_b.cpu().numpy())

                #
                # Apply the classification model
                #
                embed_a = base_model(input_a).view(input_a.size(0), -1)
                embed_b = base_model(input_b).view(input_b.size(0), -1)

                #
                # Apply the setops model.
                #
                outputs_setopt = setops_model(embed_a, embed_b)
                a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a = \
                    outputs_setopt[2:8]

                embed_a_list.append(embed_a.cpu().numpy())
                embed_b_list.append(embed_b.cpu().numpy())
                a_S_b_list.append(a_S_b.cpu().numpy())
                b_S_a_list.append(b_S_a.cpu().numpy())
                a_U_b_list.append(a_U_b.cpu().numpy())
                b_U_a_list.append(b_U_a.cpu().numpy())
                a_I_b_list.append(a_I_b.cpu().numpy())
                b_I_a_list.append(b_I_a.cpu().numpy())

                #
                # Calculate the target setops operations
                #
                target_a_list.append(target_a.cpu().numpy())
                target_b_list.append(target_b.cpu().numpy())
                target_a = target_a.type(torch.cuda.ByteTensor)
                target_b = target_b.type(torch.cuda.ByteTensor)

                target_a_I_b = target_a & target_b
                target_a_U_b = target_a | target_b
                target_a_S_b = target_a & ~target_a_I_b
                target_b_S_a = target_b & ~target_a_I_b

                target_a_I_b_list.append(target_a_I_b.type(torch.cuda.FloatTensor).cpu().numpy())
                target_a_U_b_list.append(target_a_U_b.type(torch.cuda.FloatTensor).cpu().numpy())
                target_a_S_b_list.append(target_a_S_b.type(torch.cuda.FloatTensor).cpu().numpy())
                target_b_S_a_list.append(target_b_S_a.type(torch.cuda.FloatTensor).cpu().numpy())

        ids_a_all = np.concatenate(ids_a_list, axis=0)
        ids_b_all = np.concatenate(ids_b_list, axis=0)
        del ids_a_list, ids_b_list

        a_S_b_list, b_S_a_list = [], []
        target_a_S_b_list, target_b_S_a_list = [], []
        ids_a_list, ids_b_list = [], []
        with torch.no_grad():
            for batch in tqdm(pair_loader_sub):
                input_a, input_b, target_a, target_b, id_a, id_b = _prepare_batch(batch, device=self.device)

                ids_a_list.append(id_a.cpu().numpy())
                ids_b_list.append(id_b.cpu().numpy())

                #
                # Apply the classification model
                #
                embed_a = base_model(input_a).view(input_a.size(0), -1)
                embed_b = base_model(input_b).view(input_b.size(0), -1)

                #
                # Apply the setops model.
                #
                outputs_setopt = setops_model(embed_a, embed_b)
                a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a = \
                    outputs_setopt[2:8]

                a_S_b_list.append(a_S_b.cpu().numpy())
                b_S_a_list.append(b_S_a.cpu().numpy())

                #
                # Calculate the target setops operations
                #
                target_a = target_a.type(torch.cuda.ByteTensor)
                target_b = target_b.type(torch.cuda.ByteTensor)

                target_a_I_b = target_a & target_b
                target_a_S_b = target_a & ~target_a_I_b
                target_b_S_a = target_b & ~target_a_I_b

                target_a_S_b_list.append(target_a_S_b.type(torch.cuda.FloatTensor).cpu().numpy())
                target_b_S_a_list.append(target_b_S_a.type(torch.cuda.FloatTensor).cpu().numpy())

        ids_a_sub = np.concatenate(ids_a_list, axis=0)
        ids_b_sub = np.concatenate(ids_b_list, axis=0)

        def score_outputs(output_chunk, target_chunk, ids_a_chunk, ids_b_chunk, val_labels, K=5):
            _, inds_chunk = tree.query(np.array(output_chunk), k=K+2)

            ious = []
            inds_list = []
            input_ids_list = []
            targets_list = []
            for target, inds, id_a, id_b in zip(target_chunk, inds_chunk, ids_a_chunk, ids_b_chunk):

                #
                # Verify that the target label exists in the validation dataset.
                #
                if label2hash(target) not in self.val_labels_set:
                    continue

                #
                # Verify that didn't return one of the original vectors.
                #
                inds = inds.flatten()
                ids = [val_loader.dataset.image_ids[i] for i in inds]
                banned_ids = {id_a, id_b}
                inds_ok = []
                for i, id_ in enumerate(ids):
                    if id_ in banned_ids:
                        continue
                    inds_ok.append(inds[i])

                #
                # Calculate the IOU for different k
                #
                ious_k = []
                for k in (1, 3, 5):
                    inds_k = list(inds_ok[:k])
                    ious_k.append(np.max([calc_IOU(target, val_labels[i]) for i in inds_k]))

                ious.append(ious_k)
                inds_list.append(inds_ok[:K])
                input_ids_list.append([id_a, id_b])
                targets_list.append(target)

            return ious, inds_list, input_ids_list, targets_list

        #
        # Output results
        #
        logging.info("Calculate scores.")
        results_path = Path(self.results_path)
        for outputs, targets, ids_a, ids_b, name in zip(
                (a_S_b_list, b_S_a_list, a_U_b_list, b_U_a_list, a_I_b_list, b_I_a_list, embed_a_list, embed_b_list),
                (target_a_S_b_list, target_b_S_a_list, target_a_U_b_list, target_a_U_b_list, target_a_I_b_list,
                 target_a_I_b_list, target_a_list, target_b_list),
                (ids_a_sub, ids_a_sub, ids_a_all, ids_a_all, ids_a_all, ids_a_all, ids_a_all, ids_a_all),
                (ids_b_sub, ids_b_sub, ids_b_all, ids_b_all, ids_b_all, ids_b_all, ids_b_all, ids_b_all),
                ("a_S_b", "b_S_a", "a_U_b", "b_U_a", "a_I_b", "b_I_a", "a", "b")):

            outputs = np.concatenate(outputs, axis=0)
            targets = np.concatenate(targets, axis=0)

            # res = Parallel(n_jobs=-1)(
            res = Parallel(n_jobs=1)(
                delayed(score_outputs)(output_chunk, target_chunk, ids_a_chunk, ids_b_chunk, val_labels) \
                for output_chunk, target_chunk, ids_a_chunk, ids_b_chunk in \
                zip(chunked(outputs[::self.skip_tests], 200), chunked(targets[::self.skip_tests], 200),
                    chunked(ids_a[::self.skip_tests], 200), chunked(ids_b[::self.skip_tests], 200))
            )
            ious, inds_list, input_ids_list, targets_list = list(zip(*res))

            ious = np.concatenate(ious, axis=0)
            selected_inds = np.concatenate(inds_list, axis=0)
            input_ids = np.concatenate(input_ids_list, axis=0)
            targets = np.concatenate(targets_list, axis=0)

            del inds_list, input_ids_list, targets_list

            with (results_path / "results_{}.pkl".format(name)).open("wb") as f:
                pickle.dump(dict(ious=ious, selected_inds=selected_inds, input_ids=input_ids, targets=targets), f)

            logging.info(
                'Test {} average recall (k=1, 3, 5): {}'.format(
                    name, np.mean(ious, axis=0)
                )
            )

    def embed_dataset(self, base_model, val_loader):
        """Calculate the validation embedding.

        Args:
            base_model:
            val_loader:

        Returns:

        """

        logging.info("Calculate the validation embeddings.")
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_, labels = convert_tensor(batch, device=self.device)
                if self.paper_reproduce:
                    embed = torch.tanh(base_model(input_))
                else:
                    embed = base_model(input_)

                val_outputs.append(embed.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        val_outputs = np.concatenate(val_outputs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        return val_labels, val_outputs

    def setup_datasets(self):
        """Load the training datasets."""

        logging.info("Setting up the datasets.")
        # TODO: comment out if you don't want to copy coco to /tmp/aa
        # copy_coco_data()
        CocoDatasetPairs = getattr(alfassy, "CocoDatasetPairs")
        CocoDatasetPairsSub = getattr(alfassy, "CocoDatasetPairsSub")
        if self.paper_reproduce:
            scaler = transforms.Scale((350, 350))
        else:
            scaler = transforms.Resize(self.crop_size)

        val_transform = transforms.Compose(
            [
                scaler,
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        CocoDataset = getattr(alfassy, "CocoDataset")

        val_dataset = CocoDataset(
            root_dir=self.coco_path,
            set_name='val2014',
            unseen_set=self.unseen,
            transform=val_transform,
            debug_size=self.debug_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        pair_dataset = CocoDatasetPairs(
            root_dir=self.coco_path,
            set_name='val2014',
            unseen_set=self.unseen,
            transform=val_transform,
            return_ids=True,
            debug_size=self.debug_size
        )

        pair_loader = DataLoader(
            pair_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        pair_dataset_sub = CocoDatasetPairsSub(
            root_dir=self.coco_path,
            set_name='val2014',
            unseen_set=self.unseen,
            transform=val_transform,
            return_ids=True,
            debug_size=self.debug_size
        )

        pair_loader_sub = DataLoader(
            pair_dataset_sub,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return val_loader, pair_loader, pair_loader_sub

    def setup_model(self):
        """Create or resume the models."""

        logging.info("Setup the models.")

        logging.info("{} model".format(self.base_network_name))
        models_path = Path(self.resume_path)

        if self.base_network_name.lower().startswith("resnet"):
            base_model, classifier = getattr(setops_models, self.base_network_name)(
                num_classes=80,
                avgpool_kernel=self.avgpool_kernel
            )
        else:
            base_model = Inception3(aux_logits=False, transform_input=True)
            classifier = getattr(setops_models, self.classifier_name)(num_classes=80)
            if self.init_inception:
                logging.info("Initialize inception model using paper's networks.")

                checkpoint = torch.load(models_path / 'paperBaseModel')
                base_model = Inception3(aux_logits=False, transform_input=True)
                base_model.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in base_model.state_dict()}
                )
                classifier.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in classifier.state_dict()}
                )

        setops_model_cls = getattr(setops_models, self.sets_network_name)
        setops_model = setops_model_cls(
            input_dim=2048,
            S_latent_dim=self.ops_latent_dim, S_layers_num=self.ops_layer_num,
            I_latent_dim=self.ops_latent_dim, I_layers_num=self.ops_layer_num,
            U_latent_dim=self.ops_latent_dim, U_layers_num=self.ops_layer_num,
            block_cls_name=self.sets_block_name, basic_block_cls_name=self.sets_basic_block_name,
            dropout_ratio=self.setops_dropout,
        )
        if self.resume_path:
            logging.info("Resuming the models.")
            if not self.init_inception:
                base_model.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_base_model_{}*.pth".format(self.resume_epoch)))[-1])
                )
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_classifier_{}*.pth".format(self.resume_epoch)))[-1])
                )
            if self.paper_reproduce:
                logging.info("using paper models")
                setops_model_cls = getattr(setops_models, "SetOpsModulePaper")
                setops_model = setops_model_cls(models_path)
            else:
                setops_model.load_state_dict(
                    torch.load(
                        sorted(
                            models_path.glob("networks_setops_model_{}*.pth".format(self.resume_epoch))
                        )[-1]
                    )
                )

        return base_model, classifier, setops_model


if __name__ == "__main__":
    main = Main()
    main.initialize()
    main.start()
