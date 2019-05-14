"""Calculate precision on the seen classes of COCO."""

import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from oneshot import setops_models
from oneshot import alfassy
from oneshot.coco import copy_coco_data

from experiment import Experiment

from CCC import setupCUDAdevice

setupCUDAdevice()
# PAPER_MODEL = "/dccstor/alfassy/saved_models/inception_trainCocoIncHalf642018.10.9.13:44epoch:30"
# PAPER_DISCRIMINATOR = "/dccstor/alfassy/saved_models/classifierCocoFeatureClass162018.10.16.19:10best"

from ignite._utils import convert_tensor

from traitlets import Bool, Float, Int, Unicode

setupCUDAdevice()


def _prepare_batch(batch, device=None):
    return [convert_tensor(x, device=device) for x in batch]


class Main(Experiment):

    description = Unicode(u"Calculate precision-recall accuracy of trained coco model.")

    #
    # Run setup
    #
    batch_size = Int(256, config=True, help="Batch size.")
    num_workers = Int(8, config=True, help="Number of workers to use for data loading.")
    device = Unicode("cuda", config=True, help="Use `cuda` backend.")

    #
    # Hyper parameters.
    #
    unseen = Bool(False, config=True, help="Test on unseen classes.")
    skip_tests = Int(1, config=True, help="How many test pairs to skip? for better runtime")
    debug_size = Int(-1, config=True, help="Reduce dataset sizes. This is useful when developing the script.")

    #
    # Resume previous run parameters.
    #
    resume_path = Unicode(u"/dccstor/alfassy/finalLaSO/code_release/paperModels", config=True, help="Resume from checkpoint file (requires using also '--resume_epoch'.")
    resume_epoch = Int(0, config=True, help="Epoch to resume (requires using also '--resume_path'.")
    init_inception = Bool(True, config=True, help="Initialize the inception networks using the paper's base network.")

    #
    # Network hyper parameters
    #
    base_network_name = Unicode("Inception3", config=True, help="Name of base network to use.")
    avgpool_kernel = Int(7, config=True,
                         help="Size of the last avgpool layer in the Resnet. Should match the cropsize.")
    classifier_name = Unicode("Inception3Classifier", config=True, help="Name of classifier to use.")
    sets_network_name = Unicode("SetOpsResModule", config=True, help="Name of setops module to use.")
    sets_block_name = Unicode("SetopResBlock_v1", config=True, help="Name of setops network to use.")
    sets_basic_block_name = Unicode("SetopResBasicBlock", config=True,
                                    help="Name of the basic setops block to use (where applicable).")
    ops_layer_num = Int(1, config=True, help="Ops Module layers num.")
    ops_latent_dim = Int(1024, config=True, help="Ops Module inner latent dim.")
    setops_dropout = Float(0, config=True, help="Dropout ratio of setops module.")
    crop_size = Int(224, config=True, help="Size of input crop (Resnet 224, inception 299).")
    scale_size = Int(350, config=True, help="Size of input scale for data augmentation")
    paper_reproduce = Bool(False, config=True, help="Use paper reproduction settings.")
    discriminator_name = Unicode("AmitDiscriminator", config=True, help="Name of discriminator (unseen classifier) to use.")
    embedding_dim = Int(2048, config=True, help="Dimensionality of the LaSO space.")
    classifier_latent_dim = Int(2048, config=True, help="Dimensionality of the classifier latent space.")

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
        pair_dataset, pair_loader, pair_dataset_sub, pair_loader_sub = self.setup_datasets()

        logging.info("Calcualting classifications:")
        output_a_list, output_b_list, fake_a_list, fake_b_list, target_a_list, target_b_list = [], [], [], [], [], []
        a_S_b_list, b_S_a_list, a_U_b_list, b_U_a_list, a_I_b_list, b_I_a_list = [], [], [], [], [], []
        target_a_I_b_list, target_a_U_b_list, target_a_S_b_list, target_b_S_a_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(pair_loader):
                input_a, input_b, target_a, target_b = _prepare_batch(batch, device=self.device)

                #
                # Apply the classification model
                #
                embed_a = base_model(input_a).view(input_a.size(0), -1)
                embed_b = base_model(input_b).view(input_b.size(0), -1)
                output_a = classifier(embed_a)
                output_b = classifier(embed_b)

                #
                # Apply the setops model.
                #
                outputs_setopt = setops_model(embed_a, embed_b)
                # fake_a, fake_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a = \
                #     [classifier(o) for o in outputs_setopt[:8]]
                a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a = \
                    [classifier(o) for o in outputs_setopt[2:8]]

                output_a_list.append(output_a.cpu().numpy())
                output_b_list.append(output_b.cpu().numpy())
                # fake_a_list.append(fake_a.cpu().numpy())
                # fake_b_list.append(fake_b.cpu().numpy())
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

        logging.info("Calculating classifications for subtraction independently:")
        a_S_b_list, b_S_a_list = [], []
        target_a_S_b_list, target_b_S_a_list = [], []
        with torch.no_grad():
            for batch in tqdm(pair_loader_sub):
                input_a, input_b, target_a, target_b = _prepare_batch(batch, device=self.device)

                #
                # Apply the classification model
                #
                embed_a = base_model(input_a).view(input_a.size(0), -1)
                embed_b = base_model(input_b).view(input_b.size(0), -1)
                #
                # Apply the setops model.
                #
                outputs_setopt = setops_model(embed_a, embed_b)
                a_S_b, b_S_a, _, _, _, _ = \
                    [classifier(o) for o in outputs_setopt[2:8]]

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

        #
        # Output restuls
        #
        logging.info("Calculating precision:")
        for output, target, name in tqdm(zip(
                (output_a_list, output_b_list, a_S_b_list, b_S_a_list, a_U_b_list, b_U_a_list, a_I_b_list, b_I_a_list),
                (target_a_list, target_b_list, target_a_S_b_list, target_b_S_a_list, target_a_U_b_list,
                 target_a_U_b_list, target_a_I_b_list, target_a_I_b_list),
                ("real_a", "real_b", "a_S_b", "b_S_a", "a_U_b", "b_U_a", "a_I_b", "b_I_a"))):

            output = np.concatenate(output, axis=0)
            target = np.concatenate(target, axis=0)

            ap = [average_precision_score(target[:, i], output[:, i]) for i in range(output.shape[1])]
            pr_graphs = [precision_recall_curve(target[:, i], output[:, i]) for i in range(output.shape[1])]
            ap_sum = 0
            for label in pair_dataset.labels_list:
                ap_sum += ap[label]
            ap_avg = ap_sum / len(pair_dataset.labels_list)
            logging.info(
                'Test {} average precision score, macro-averaged over all {} classes: {}'.format(
                    name, len(pair_dataset.labels_list), ap_avg)
            )

            with open(os.path.join(self.results_path, "{}_results.pkl".format(name)), "wb") as f:
                pickle.dump(dict(ap=ap, pr_graphs=pr_graphs), f)

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
            base_model = setops_models.Inception3(aux_logits=False, transform_input=True)
            classifier = getattr(setops_models, self.classifier_name)(num_classes=80)
            if self.init_inception:
                logging.info("Initialize inception model using paper's networks.")
                checkpoint = torch.load(models_path / 'paperBaseModel')
                base_model = setops_models.Inception3(aux_logits=False, transform_input=True)
                base_model.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in base_model.state_dict()}
                )
                classifier.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in classifier.state_dict()}
                )
        setops_model_cls = getattr(setops_models, self.sets_network_name)
        setops_model = setops_model_cls(
            input_dim=self.embedding_dim,
            S_latent_dim=self.ops_latent_dim, S_layers_num=self.ops_layer_num,
            I_latent_dim=self.ops_latent_dim, I_layers_num=self.ops_layer_num,
            U_latent_dim=self.ops_latent_dim, U_layers_num=self.ops_layer_num,
            block_cls_name=self.sets_block_name, basic_block_cls_name=self.sets_basic_block_name,
            dropout_ratio=self.setops_dropout,
        )

        if self.unseen:
            #
            # In the unseen mode, we have to load the trained discriminator.
            #
            discriminator_cls = getattr(setops_models, self.discriminator_name)
            classifier = discriminator_cls(
                input_dim=self.embedding_dim,
                latent_dim=self.classifier_latent_dim
            )

        if not self.resume_path:
            raise FileNotFoundError("resume_path is compulsory in test_precision")
        logging.info("Resuming the models.")
        if not self.init_inception:
            base_model.load_state_dict(
                torch.load(sorted(models_path.glob("networks_base_model_{}*.pth".format(self.resume_epoch)))[-1])
            )

        if self.paper_reproduce:
            logging.info("using paper models")
            setops_model_cls = getattr(setops_models, "SetOpsModulePaper")
            setops_model = setops_model_cls(models_path)
            if self.unseen:
                checkpoint = torch.load(models_path / 'paperDiscriminator')
                classifier.load_state_dict(checkpoint['state_dict'])
        else:
            setops_model.load_state_dict(
                torch.load(
                    sorted(
                        models_path.glob("networks_setops_model_{}*.pth".format(self.resume_epoch))
                    )[-1]
                )
            )
            if self.unseen:
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_discriminator_{}*.pth".format(self.resume_epoch)))[-1])
                )
            elif not self.init_inception:
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_classifier_{}*.pth".format(self.resume_epoch)))[-1])
                )

        return base_model, classifier, setops_model

    def setup_datasets(self):
        """Load the training datasets."""

        copy_coco_data()

        logging.info("Setting up the datasets.")
        CocoDatasetPairs = getattr(alfassy, "CocoDatasetPairs")
        CocoDatasetPairsSub = getattr(alfassy, "CocoDatasetPairsSub")
        if self.paper_reproduce:
            logging.info("Setting up the datasets and augmentation for paper reproduction")
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
        pair_dataset = CocoDatasetPairs(
            root_dir="/tmp/aa/coco",
            set_name='val2014',
            unseen_set=self.unseen,
            transform=val_transform,
            debug_size=self.debug_size
        )

        pair_loader = DataLoader(
            pair_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        pair_dataset_sub = CocoDatasetPairsSub(
            root_dir="/tmp/aa/coco",
            set_name='val2014',
            unseen_set=self.unseen,
            transform=val_transform,
            debug_size=self.debug_size
        )

        pair_loader_sub = DataLoader(
            pair_dataset_sub,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return pair_dataset, pair_loader, pair_dataset_sub, pair_loader_sub


if __name__ == "__main__":
    main = Main()
    main.initialize()
    main.start()
