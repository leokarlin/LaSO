"""Train the set-operations models on the COCO dataset.

"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import logging
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
torch.backends.cudnn.benchmark = True

from oneshot.coco import copy_coco_data
from oneshot.alfassy import CocoDatasetPairs
from oneshot.alfassy import labels_list_to_1hot
from oneshot.ignite.metrics import EWMeanSquaredError
from oneshot.ignite.metrics import mAP
from oneshot.ignite.metrics import MultiLabelSoftMarginIOUaccuracy
from oneshot.pytorch import FocalLoss
from oneshot import setops_models
from oneshot.setops_models imporgt Inception3
from oneshot.utils import conditional

from experiment import MLflowExperiment
# from experiment import TensorboardXExperiment
# from experiment import VisdomExperiment

from oneshot.utils import setupCUDAdevice

from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from ignite.contrib.handlers import MlflowLogger
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import LinearCyclicalScheduler
from ignite.contrib.handlers import ConcatScheduler
from ignite.contrib.handlers import ReduceLROnPlateau
# from ignite.contrib.handlers import TensorboardLogger
from ignite._utils import convert_tensor

from traitlets import Bool, Enum, Int, Float, Unicode

# setupCUDAdevice()

LOG_INTERVAL = 10
CKPT_PREFIX = 'networks'

#
# Seed the random states
#
np.random.seed(0)
random_state = np.random.RandomState(0)

#import warnings
#warnings.filterwarnings("error")


def _prepare_batch(batch, device=None):
    return [convert_tensor(x, device=device) for x in batch]


def create_setops_trainer(
        base_model,
        classifier,
        setops_model,
        optimizer,
        criterion1,
        criterion2,
        params_object,
        metrics={},
        device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        base_model.to(device)
        classifier.to(device)
        setops_model.to(device)

    def _update(engine, batch):

        if params_object.train_base:
            base_model.train()
        else:
            base_model.eval()

        classifier.train()
        setops_model.train()

        optimizer.zero_grad()

        input_a, input_b, target_a, target_b = _prepare_batch(batch, device=device)

        #
        # Apply the classification model
        #
        with conditional(not params_object.train_base, torch.no_grad()):
            embed_a = base_model(input_a)
            embed_b = base_model(input_b)

        output_a = classifier(embed_a)
        output_b = classifier(embed_b)

        #
        # Apply the setopt model.
        #
        outputs_setopt = setops_model(embed_a, embed_b)
        fake_a, fake_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
        a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
        a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a = \
                    [classifier(o) for o in outputs_setopt]
        fake_a_em, fake_b_em, a_S_b_em, b_S_a_em, a_U_b_em, b_U_a_em, a_I_b_em, b_I_a_em, \
        a_S_b_b_em, b_S_a_a_em, a_I_b_b_em, b_I_a_a_em, a_U_b_b_em, b_U_a_a_em, \
        a_S_b_I_a_em, b_S_a_I_b_em, a_S_a_I_b_em, b_S_b_I_a_em = outputs_setopt

        loss_class = criterion1(output_a, target_a) + criterion1(output_b, target_b)
        loss_class_out = criterion1(fake_a, target_a) + criterion1(fake_b, target_b)
        if params_object.mc_toggle:
            loss_recon = criterion2(embed_a, fake_a_em) + criterion2(embed_b, fake_b_em)
            return_loss_recon = loss_recon.item()
        else:
            loss_recon = 0
            return_loss_recon = 0

        #
        # Calculate the target setopt operations
        #
        target_a = target_a.type(torch.cuda.ByteTensor)
        target_b = target_b.type(torch.cuda.ByteTensor)

        target_a_I_b = target_a & target_b
        target_a_U_b = target_a | target_b
        target_a_S_b = target_a & ~target_a_I_b
        target_b_S_a = target_b & ~target_a_I_b

        target_a_I_b = target_a_I_b.type(torch.cuda.FloatTensor)
        target_a_U_b = target_a_U_b.type(torch.cuda.FloatTensor)
        target_a_S_b = target_a_S_b.type(torch.cuda.FloatTensor)
        target_b_S_a = target_b_S_a.type(torch.cuda.FloatTensor)

        loss_class_S = criterion1(a_S_b, target_a_S_b) + criterion1(b_S_a, target_b_S_a)
        loss_class_U = criterion1(a_U_b, target_a_U_b)
        loss_class_I = criterion1(a_I_b, target_a_I_b)
        if params_object.tautology_class_toggle:
            loss_class_S += criterion1(a_S_b_b, target_a_S_b) + criterion1(b_S_a_a, target_b_S_a)
            loss_class_S += criterion1(a_S_a_I_b, target_a_S_b) + criterion1(b_S_a_I_b, target_b_S_a) +\
                            criterion1(b_S_b_I_a, target_b_S_a) + criterion1(a_S_b_I_a, target_a_S_b)
            loss_class_U += criterion1(a_U_b_b, target_a_U_b) + criterion1(b_U_a_a, target_a_U_b)
            loss_class_I += criterion1(a_I_b_b, target_a_I_b) + criterion1(b_I_a_a, target_a_I_b)

        if params_object.tautology_recon_toggle:
            loss_recon_S = criterion2(a_S_b_em, a_S_b_b_em) + criterion2(a_S_b_em, a_S_a_I_b_em) + \
                           criterion2(a_S_b_em, a_S_b_I_a_em)
            loss_recon_S += criterion2(b_S_a_em, b_S_a_a_em) + criterion2(b_S_a_em, b_S_a_I_b_em) + \
                            criterion2(b_S_a_em, b_S_b_I_a_em)
            return_recon_S = loss_recon_S.item()
        else:
            loss_recon_S = 0
            return_recon_S = 0

        if params_object.sym_class_toggle:
            loss_class_U += criterion1(b_U_a, target_a_U_b)
            loss_class_I += criterion1(b_I_a, target_a_I_b)

        if params_object.sym_recon_toggle:
            loss_recon_U = criterion2(a_U_b_em, b_U_a_em)
            loss_recon_I = criterion2(a_I_b_em, b_I_a_em)
            return_recon_U = loss_recon_U.item()
            return_recon_I = loss_recon_I.item()
        else:
            loss_recon_U = 0
            loss_recon_I = 0
            return_recon_U = 0
            return_recon_I = 0

        loss = loss_class
        loss += 0 if params_object.class_fake_loss_weight == 0 else params_object.class_fake_loss_weight * loss_class_out
        loss += 0 if (params_object.recon_loss_weight == 0) or (not loss_recon) else params_object.recon_loss_weight * loss_recon
        loss += 0 if params_object.class_S_loss_weight == 0 else params_object.class_S_loss_weight * loss_class_S
        loss += 0 if (params_object.recon_loss_weight == 0) or (not loss_recon_I) else params_object.recon_loss_weight * loss_recon_S
        loss += 0 if params_object.class_U_loss_weight == 0 else params_object.class_U_loss_weight * loss_class_U
        loss += 0 if (params_object.recon_loss_weight == 0) or (not loss_recon_U) else params_object.recon_loss_weight * loss_recon_U
        loss += 0 if params_object.class_I_loss_weight == 0 else params_object.class_I_loss_weight * loss_class_I
        loss += 0 if (params_object.recon_loss_weight == 0) or (not loss_recon_I) else params_object.recon_loss_weight * loss_recon_I

        loss.backward()
        optimizer.step()

        return {
            "main": loss.item(),
            "real class": loss_class.item(),
            "fake class": loss_class_out.item(),
            "fake MSE": return_loss_recon,
            "S MSE": return_recon_S,
            "U MSE": return_recon_U,
            "I MSE": return_recon_I,
            "S class": loss_class_S.item(),
            "U class": loss_class_U.item(),
            "I class": loss_class_I.item()
        }

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_setops_evaluator(
        base_model,
        classifier,
        setops_model,
        metrics={},
        device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        base_model.to(device)
        classifier.to(device)
        setops_model.to(device)

    def _inference(engine, batch):

        base_model.eval()
        classifier.eval()
        setops_model.eval()

        with torch.no_grad():
            input_a, input_b, target_a, target_b = _prepare_batch(batch, device=device)

            #
            # Apply the classification model
            #
            embed_a = base_model(input_a)
            output_a = classifier(embed_a)
            embed_b = base_model(input_b)
            output_b = classifier(embed_b)

            #
            # Apply the setops model.
            #
            outputs_setopt = setops_model(embed_a, embed_b)
            fake_a, fake_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
            a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
            a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a = \
                [classifier(o) for o in outputs_setopt]
            fake_a_em, fake_b_em = outputs_setopt[:2]

            #
            # Calculate the target setops operations
            #
            target_a_bt = target_a.type(torch.cuda.ByteTensor)
            target_b_bt = target_b.type(torch.cuda.ByteTensor)

            target_a_I_b = target_a_bt & target_b_bt
            target_a_U_b = target_a_bt | target_b_bt
            target_a_S_b = target_a_bt & ~target_a_I_b
            target_b_S_a = target_b_bt & ~target_a_I_b

            target_a_I_b = target_a_I_b.type(torch.cuda.FloatTensor)
            target_a_U_b = target_a_U_b.type(torch.cuda.FloatTensor)
            target_a_S_b = target_a_S_b.type(torch.cuda.FloatTensor)
            target_b_S_a = target_b_S_a.type(torch.cuda.FloatTensor)

            return dict(
                outputs={
                    "real class a": output_a,
                    "real class b": output_b,
                    "fake class a": fake_a,
                    "fake class b": fake_b,
                    "a_S_b class": a_S_b,
                    "b_S_a class": b_S_a,
                    "a_U_b class": a_U_b,
                    "b_U_a class": b_U_a,
                    "a_I_b class": a_I_b,
                    "b_I_a class": b_I_a,
                    "fake embed a": fake_a_em,
                    "fake embed b": fake_b_em,
                },
                targets={
                    "class a": target_a,
                    "class b": target_b,
                    "a_S_b class": target_a_S_b,
                    "b_S_a class": target_b_S_a,
                    "a_U_b class": target_a_U_b,
                    "a_I_b class": target_a_I_b,
                    "embed a": embed_a,
                    "embed b": embed_b,
                }
            )

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Main(MLflowExperiment):
    #
    # Resume previous run parameters.
    #
    resume_path = Unicode(u"/dccstor/faceid/results/train_coco_resnet/0198_968f3cd/1174695/190117_081837/", config=True, help="Resume from checkpoint file (requires using also '--resume_epoch'.")
    resume_epoch = Int(49, config=True, help="Epoch to resume (requires using also '--resume_path'.")
    coco_path = Unicode(u"/tmp/aa/coco", config=True, help="path to local coco dataset path")
    init_inception = Bool(False, config=True, help="Initialize the inception networks using ALFASSY's network.")

    #
    # Network hyper parameters
    #
    base_network_name = Unicode("resnet50", config=True, help="Name of base network to use.")
    avgpool_kernel = Int(7, config=True,
                         help="Size of the last avgpool layer in the Resnet. Should match the cropsize.")
    classifier_name = Unicode("Inception3Classifier", config=True, help="Name of classifier to use.")
    sets_network_name = Unicode("SetOpsResModule", config=True, help="Name of setops module to use.")
    sets_block_name = Unicode("SetopResBlock_v1", config=True, help="Name of setops network to use.")
    sets_basic_block_name = Unicode("SetopResBasicBlock", config=True,
                                    help="Name of the basic setops block to use (where applicable).")
    ops_layer_num = Int(1, config=True, help="Ops Module layer num.")
    ops_latent_dim = Int(8092, config=True, help="Ops Module latent dim.")
    setops_dropout = Float(0, config=True, help="Dropout ratio of setops module.")
    crop_size = Int(224, config=True, help="Size of input crop (Resnet 224, inception 299).")

    #
    # Run setup
    #
    batch_size = Int(16, config=True, help="Batch size.")
    num_workers = Int(8, config=True, help="Number of workers to use for data loading.")
    device = Unicode("cuda", config=True, help="Use `cuda` backend.")

    #
    # Training hyper parameters.
    #
    random_angle = Float(10, config=True, help="Angle of random augmentation.")
    random_scale = Float(0.3, config=True, help="Scale of radnom augmentation.")
    train_base = Bool(True, config=True, help="Whether to train also the base model.").tag(parameter=True)
    train_classifier = Bool(False, config=True, help="Whether to train also the classifier.")
    epochs = Int(50, config=True, help="Number of epochs to run.")
    optimizer_cls = Unicode("SGD", config=True, help="Type of optimizer to use.")
    focal_loss = Bool(False, config=True, help="Use Focal Loss.")
    recon_loss = Enum(("mse", "l1"), config=True, default_value="mse",
                      help="Type of reconstruction (embedding) loss: mse/l1.")

    lr1 = Float(0.0001, config=True, help="Learning rate start.")
    lr2 = Float(0.002, config=True, help="Learning rate end.")
    warmup_epochs = Int(3, config=True, help="Length (in epochs) of the LR warmup.")

    weight_decay = Float(0.0001, config=True, help="Weight decay (L2 regularization).").tag(parameter=True)
    recon_loss_weight = Float(1., config=True, help="Weight of reconstruction (embedding) loss.").tag(parameter=True)
    class_fake_loss_weight = Float(1., config=True, help="Weight of fake classification loss.").tag(parameter=True)
    class_S_loss_weight = Float(1., config=True, help="Weight of Substraction classification loss.").tag(parameter=True)
    class_U_loss_weight = Float(1., config=True, help="Weight of Union classification loss.").tag(parameter=True)
    class_I_loss_weight = Float(1., config=True, help="Weight of Intersection classification loss.").tag(parameter=True)
    # loss
    sym_class_toggle = Bool(True, config=True, help="Should we use symmetric classification loss?")
    sym_recon_toggle = Bool(True, config=True, help="Should we use symmetric reconstruction loss?")
    mc_toggle = Bool(True, config=True, help="Should we use anti mode collapse loss?")
    tautology_recon_toggle = Bool(True, config=True, help="Should we use tautology reconstruction loss?")
    tautology_class_toggle = Bool(True, config=True, help="Should we use tautology classification loss?")
    dataset_size_ratio = Int(4, config=True, help="Multiplier of training dataset.").tag(parameter=True)

    def run(self):
        # TODO: comment out if you don't want to copy coco to /tmp/aa
        # copy_coco_data()

        #
        # create model
        #
        base_model, classifier, setops_model = self.setup_model()

        #
        # Create ignite trainers and evalators.
        # Note:
        # I use "two" evaluators, the first is used for evaluating the model on the training data.
        # This separation is done so as that checkpoint will be done according to the results of
        # the validation evaluator.
        #
        trainer, train_loader = self.setup_training(
            base_model,
            classifier,
            setops_model
        )

        #
        # kick everything off
        #
        trainer.run(
            train_loader,
            max_epochs=self.epochs
        )

    def setup_training(
            self,
            base_model,
            classifier,
            setops_model):

        #
        # Create the train and test dataset.
        #
        train_loader, train_subset_loader, val_loader = self.setup_datasets()

        logging.info("Setup logging and controls.")

        #
        # Setup metrics plotters.
        #
        mlflow_logger = MlflowLogger()

        #
        # Setup the optimizer.
        #
        logging.info("Setup optimizers and losses.")

        parameters = list(base_model.parameters())
        parameters += list(setops_model.parameters())
        if self.train_classifier:
            parameters += list(classifier.parameters())

        if self.optimizer_cls == "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.lr1,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.lr1,
                weight_decay=self.weight_decay
            )

        if self.focal_loss:
            attr_loss = FocalLoss().cuda()
        else:
            attr_loss = torch.nn.MultiLabelSoftMarginLoss().cuda()

        recon_loss = torch.nn.MSELoss() if self.recon_loss == "mse" else torch.nn.L1Loss()

        #
        # Setup the trainer object and its logging.
        #
        logging.info("Setup trainer")
        trainer = create_setops_trainer(
            base_model,
            classifier,
            setops_model,
            optimizer,
            criterion1=attr_loss,
            criterion2=recon_loss.cuda(),
            params_object=self,
            device=self.device
        )
        ProgressBar(bar_format=None).attach(trainer)

        mlflow_logger.attach(
            engine=trainer,
            prefix="Train ",
            plot_event=Events.ITERATION_COMPLETED,
            update_period=LOG_INTERVAL,
            output_transform=lambda x: x
        )

        #
        # Define the evaluation metrics.
        #
        logging.info("Setup evaluator")
        evaluation_losses = {
            'real class loss':
                Loss(torch.nn.MultiLabelSoftMarginLoss().cuda(), lambda o: (o["outputs"]["real class a"], o["targets"]["class a"])) + \
                Loss(torch.nn.MultiLabelSoftMarginLoss().cuda(), lambda o: (o["outputs"]["real class b"], o["targets"]["class b"])),
            'fake class loss':
                Loss(torch.nn.MultiLabelSoftMarginLoss().cuda(), lambda o: (o["outputs"]["fake class a"], o["targets"]["class a"])) + \
                Loss(torch.nn.MultiLabelSoftMarginLoss().cuda(), lambda o: (o["outputs"]["fake class b"], o["targets"]["class b"])),
            '{} fake loss'.format(self.recon_loss):
                (Loss(recon_loss.cuda(), lambda o: (o["outputs"]["fake embed a"], o["targets"]["embed a"])) +
                Loss(recon_loss.cuda(), lambda o: (o["outputs"]["fake embed b"], o["targets"]["embed b"]))) / 2,
        }
        labels_list = train_loader.dataset.labels_list
        mask = labels_list_to_1hot(labels_list, labels_list).astype(np.bool)
        evaluation_accuracies = {
            'real class acc':
                (MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["real class a"], o["targets"]["class a"])) +
                MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["real class b"], o["targets"]["class b"]))) / 2,
            'fake class acc':
                (MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["fake class a"], o["targets"]["class a"])) +
                MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["fake class b"], o["targets"]["class b"]))) / 2,
            'S class acc':
                (MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["a_S_b class"], o["targets"]["a_S_b class"])) +
                MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["b_S_a class"], o["targets"]["b_S_a class"]))) / 2,
            'I class acc':
                (MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["a_I_b class"], o["targets"]["a_I_b class"])) +
                MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["b_I_a class"], o["targets"]["a_I_b class"]))) / 2,
            'U class acc':
                (MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["a_U_b class"], o["targets"]["a_U_b class"])) +
                MultiLabelSoftMarginIOUaccuracy(lambda o: (o["outputs"]["b_U_a class"], o["targets"]["a_U_b class"]))) / 2,
            'MSE fake acc':
                (EWMeanSquaredError(lambda o: (o["outputs"]["fake embed a"], o["targets"]["embed a"])) +
                EWMeanSquaredError(lambda o: (o["outputs"]["fake embed b"], o["targets"]["embed b"]))) / 2,
            'real mAP': mAP(mask=mask,
                            output_transform=lambda o: (o["outputs"]["real class a"], o["targets"]["class a"])),
            'fake mAP': mAP(mask=mask,
                            output_transform=lambda o: (o["outputs"]["fake class a"], o["targets"]["class a"])),
            'S mAP': mAP(mask=mask,
                         output_transform=lambda o: (o["outputs"]["a_S_b class"], o["targets"]["a_S_b class"])),
            'I mAP': mAP(mask=mask,
                         output_transform=lambda o: (o["outputs"]["a_I_b class"], o["targets"]["a_I_b class"])),
            'U mAP': mAP(mask=mask,
                         output_transform=lambda o: (o["outputs"]["a_U_b class"], o["targets"]["a_U_b class"])),
        }

        #
        # Setup the training evaluator object and its logging.
        #
        train_evaluator = create_setops_evaluator(
            base_model,
            classifier,
            setops_model,
            metrics=evaluation_accuracies.copy(),
            device=self.device
        )

        mlflow_logger.attach(
            engine=train_evaluator,
            prefix="Train Eval ",
            plot_event=Events.EPOCH_COMPLETED,
            metric_names=list(evaluation_accuracies.keys())
        )
        ProgressBar(bar_format=None).attach(train_evaluator)


        #
        # Setup the evaluator object and its logging.
        #
        evaluator = create_setops_evaluator(
            base_model,
            classifier,
            setops_model,
            metrics={**evaluation_losses, **evaluation_accuracies},
            device=self.device
        )

        mlflow_logger.attach(
            engine=evaluator,
            prefix="Eval ",
            plot_event=Events.EPOCH_COMPLETED,
            metric_names=list({**evaluation_losses, **evaluation_accuracies}.keys())
        )
        ProgressBar(bar_format=None).attach(evaluator)

        #
        # Checkpoint of the model
        #
        self.setup_checkpoint(base_model, classifier, setops_model, evaluator)

        logging.info("Setup schedulers.")

        #
        # Update learning rate manually using the Visdom interface.
        #
        one_cycle_size = len(train_loader) * self.warmup_epochs * 2

        scheduler_1 = LinearCyclicalScheduler(
            optimizer,
            "lr",
            start_value=self.lr1,
            end_value=self.lr2,
            cycle_size=one_cycle_size
        )
        scheduler_2 = ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=4*len(train_loader),
            cooldown=len(train_loader),
            output_transform=lambda x: x["main"]
        )
        lr_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[one_cycle_size // 2],
                                       save_history=True)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

        #
        # Evaluation
        #
        @trainer.on(Events.EPOCH_COMPLETED)
        def epoch_completed(engine):
            #
            # Re-randomize the indices of the training dataset.
            #
            train_loader.dataset.calc_indices()

            #
            # Run the evaluator on a subset of the training dataset.
            #
            logging.info("Evaluation on a subset of the training data.")
            train_evaluator.run(train_subset_loader)

            #
            # Run the evaluator on the validation set.
            #
            logging.info("Evaluation on the eval data.")
            evaluator.run(val_loader)

        return trainer, train_loader

    def setup_checkpoint(self, base_model, classifier, setops_model, evaluator):
        """Save checkpoints of the models."""

        checkpoint_handler_acc = ModelCheckpoint(
            self.results_path,
            CKPT_PREFIX,
            score_function=lambda eng: round(
                (eng.state.metrics["fake class acc"] + eng.state.metrics["S class acc"] +
                 eng.state.metrics["I class acc"] + eng.state.metrics["U class acc"]) / 4,
                3
            ),
            score_name="val_acc",
            n_saved=2,
            require_empty=False
        )
        checkpoint_handler_last = ModelCheckpoint(
            self.results_path,
            CKPT_PREFIX,
            save_interval=2,
            n_saved=2,
            require_empty=False
        )
        evaluator.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=checkpoint_handler_acc,
            to_save={
                'base_model': base_model.state_dict(),
                'classifier': classifier.state_dict(),
                'setops_model': setops_model.state_dict(),
            }
        )
        evaluator.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=checkpoint_handler_last,
            to_save={
                'base_model': base_model.state_dict(),
                'classifier': classifier.state_dict(),
                'setops_model': setops_model.state_dict(),
            }
        )

    def setup_model(self):
        """Create or resume the models."""

        logging.info("Setup the models.")

        logging.info("{} model".format(self.base_network_name))
        if self.base_network_name.lower().startswith("resnet"):
            base_model, classifier = getattr(setops_models, self.base_network_name)(
                num_classes=80,
                avgpool_kernel=self.avgpool_kernel
            )
        else:
            base_model = getattr(setops_models, self.base_network_name)()
            classifier = getattr(setops_models, self.classifier_name)(num_classes=80)

            if self.init_inception:
                logging.info("Initialize inception model using Amit's networks.")

                checkpoint = torch.load(self.resume_path)

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
            models_path = Path(self.resume_path)
            if self.base_network_name.lower().startswith("resnet"):
                base_model.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_base_model_{}*.pth".format(self.resume_epoch)))[-1])
                )
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_classifier_{}*.pth".format(self.resume_epoch)))[-1])
                )

            setops_models_paths = sorted(models_path.glob("networks_setops_model_{}*.pth".format(self.resume_epoch)))
            if len(setops_models_paths) > 0:
                setops_model.load_state_dict(
                    torch.load(setops_models_paths[-1]).state_dict()
                )

        return base_model, classifier, setops_model

    def setup_datasets(self):
        """Load the training datasets."""

        train_transform = transforms.Compose(
            [
                transforms.Resize(self.crop_size),
                transforms.RandomRotation(degrees=self.random_angle, resample=Image.BILINEAR),
                transforms.RandomResizedCrop(
                    size=self.crop_size, scale=(1-self.random_scale, 1+self.random_scale), ratio=(1, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(self.crop_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        train_dataset = CocoDatasetPairs(
            root_dir=self.coco_path,
            set_name='train2014',
            transform=train_transform,
            dataset_size_ratio=self.dataset_size_ratio
        )
        train_subset_dataset = Subset(train_dataset, range(0, len(train_dataset), 5*self.dataset_size_ratio))
        val_dataset = CocoDatasetPairs(
            root_dir=self.coco_path,
            set_name='val2014',
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        train_subset_loader = DataLoader(
            train_subset_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return train_loader, train_subset_loader, val_loader


if __name__ == "__main__":
    main = Main()
    main.initialize()
    main.start()
