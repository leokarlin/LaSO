import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
import torch.optim
import datetime
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch
from pathlib import Path
import random
from oneshot import setops_models
from oneshot.alfassy import set_subtraction_operation, set_union_operation, set_intersection_operation
from oneshot.alfassy import configure_logging, save_checkpoint, get_learning_rate
from oneshot.alfassy import CocoDatasetAugmentation, labels_list_to_1hot
from oneshot.alfassy import IOU_real_vectors_accuracy, precision_recall_statistics

'''
training a feature vectors classifier for few shot using LaSO as augmentation.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.01, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=2048, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=80, help='number of classes for dataset')
parser.add_argument('--env_name', type=str, default='FewShotLaSO', help='env name for file naming')
parser.add_argument('--results_path', default='/dccstor/alfassy/saved_models/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: /dccstor/alfassy/saved_models/)')
parser.add_argument('--coco_path', default='/dccstor/leonidka1/data/coco', type=str, metavar='PATH',
                    help='path to the coco data folder(default:/dccstor/leonidka1/data/coco)')
parser.add_argument('--class_cap', default=5, type=int, metavar='N',
                    help='how much of the real train data should we use for training? 1 shot? 5 shot?')
parser.add_argument('--fake_limit', default=11111111, type=int, metavar='N',
                    help='How many fake vectors should we generate?')
parser.add_argument('--used_ind_path', type=str,
                    default='/dccstor/alfassy/saved_models/FVClassAug5AVG2018.10.30.15:32usedIndicesRun4.pkl',
                    help='path for the list the indices used in this tests baseline')
parser.add_argument('--class_ind_dict_path', type=str,
                    default='/dccstor/alfassy/saved_models/FVClassAug5AVG2018.10.30.15:32ClassIdxDict16Run4.pkl',
                    help='path of the dict from class name to idx, which was used in this tests baseline')
parser.add_argument('--resume_path', type=str,
                    default='/dccstor/alfassy/finalLaSO/code_release/trainedModels', help="Resume from checkpoint file")
parser.add_argument('--crop_size', default=299, type=int, metavar='N',
                    help='Size of input crop (Resnet 224, inception 299).')


# generatorCocoInter64Features2018.10.15.12:47epoch:0
# generatorCocoUnion64Features2018.10.15.12:44epoch:0

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

env_name = opt.env_name
now = datetime.datetime.now()
vis_file_out = str(env_name) + str(now.year) + '.' + str(now.month) + '.' + \
               str(now.day) + '.' + str(now.hour) + ':' + str(now.minute)
log_filename = '/dccstor/alfassy/setoper/logs/' + str(vis_file_out) + '.log'
logger = configure_logging(log_filename)
logger.info(opt)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def setup_model():
    """Create or resume the models."""
    if not opt.resume_path:
        raise FileNotFoundError("resume_path is compulsory in test_augmentation")

    models_path = Path(opt.resume_path)

    classifier = getattr(setops_models, 'Inception3Classifier')(num_classes=80)
    checkpoint = torch.load(models_path / 'paperBaseModel')
    base_model = setops_models.Inception3(aux_logits=False, transform_input=True)
    base_model.load_state_dict(
        {k: v for k, v in checkpoint["state_dict"].items() if k in base_model.state_dict()}
    )
    classifier.load_state_dict(
        {k: v for k, v in checkpoint["state_dict"].items() if k in classifier.state_dict()}
    )

    setops_model_cls = getattr(setops_models, "SetOpsModulePaper")
    setops_model = setops_model_cls(models_path)

    return base_model, classifier, setops_model


# --------------
#  Classifier
# --------------
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.BatchNorm1d(opt.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.BatchNorm1d(opt.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.BatchNorm1d(opt.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output layer
        self.aux_layer = nn.Sequential(nn.Linear(opt.latent_dim, opt.n_classes))

    def forward(self, feature_vec):
        out = self.linear_block(feature_vec)
        scores = self.aux_layer(out)
        return scores


# Loss functions
auxiliary_loss = torch.nn.BCEWithLogitsLoss()
# Load pre trained generator
base_model, classifier, setops_model = setup_model()

base_model.cuda()
classifier.cuda()
setops_model.cuda()

base_model.eval()
classifier.eval()
setops_model.eval()

# Initialize classifier
classifier = Classifier()
classifier.apply(weights_init_normal)

if cuda:
    print("using gpu")
    classifier.cuda()
    auxiliary_loss.cuda()

# ----------
#  Configure dataset
# ----------
scaler = transforms.Scale((350, 350))
val_transform = transforms.Compose(
    [
        scaler,
        transforms.CenterCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
trainODDataset = CocoDatasetAugmentation(opt.coco_path, opt.class_cap, opt.fake_limit, opt.batch_size,
                                         opt.used_ind_path, opt.class_ind_dict_path, set_name='train2014',
                                         transform=val_transform)
print("Dataset classList: ", trainODDataset.classList)
valODDataset = CocoDatasetAugmentation(opt.coco_path, opt.class_cap, opt.fake_limit, opt.batch_size, opt.used_ind_path,
                                       opt.class_ind_dict_path, set_name='val2014',
                                       transform=val_transform)

# create train and test data loaders
train_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=opt.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(valODDataset, batch_size=opt.batch_size, shuffle=False)

# setup optimizer and scheduler
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

best_test_acc = 0
# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    print("in training")
    for i, (realImages, realLabels) in enumerate(train_loader):
        batches_done = epoch * len(train_loader) + i
        batch_size = realLabels.shape[0]
        n_classes = realLabels.shape[1]
        realImages = realImages.type(FloatTensor)
        realVecs = base_model(realImages)
        realVecs = realVecs.type(FloatTensor)
        realLabels = realLabels.type(FloatTensor)
        # sample random indices for fake vectors.

        fakeIndices = random.sample(range(trainODDataset.fakeCount), trainODDataset.fakeBatchSize)
        optimizer_C.zero_grad()
        classifier.zero_grad()
        setops_model.zero_grad()
        # create a manual batch of fake vectors, done for the sake of separation.
        for fakeI, idx in enumerate(fakeIndices):
            # load a pair of 2 fake vectors and their labels
            pair = trainODDataset.fakeVectorsPairs[idx]
            labels1 = trainODDataset.load_annotations(pair[0])
            labels1 = list(set(labels1))
            labels1 = labels_list_to_1hot(labels1, trainODDataset.classList)
            labels1 = torch.from_numpy(labels1)
            labels1 = labels1.view(1, len(labels1))
            labels2 = trainODDataset.load_annotations(pair[1])
            labels2 = list(set(labels2))
            labels2 = labels_list_to_1hot(labels2, trainODDataset.classList)
            labels2 = torch.from_numpy(labels2)
            labels2 = labels2.view(1, len(labels2))
            img1 = trainODDataset.load_image(pair[0])
            img2 = trainODDataset.load_image(pair[1])
            img1 = trainODDataset.transform(img1)
            img2 = trainODDataset.transform(img2)
            img1 = img1.type(FloatTensor)
            img2 = img2.type(FloatTensor)
            img1 = img1.view(1, img1.shape[0], img1.shape[1], img1.shape[2])
            # featureVec1 = base_model(img1)
            img2 = img2.view(1, img2.shape[0], img2.shape[1], img2.shape[2])
            # featureVec2 = base_model(img2)
            # featureVec1 = featureVec1.type(FloatTensor)
            # featureVec2 = trainODDataset.img2vec.get_vec(img2, tensor=True)
            # featureVec2 = featureVec2.view(1, trainODDataset.img2vec.layer_output_size)
            # featureVec2 = featureVec2.type(FloatTensor)
            # featureVecs = torch.cat((featureVec1, featureVec2), 1)
            # choose the set operation respectively to the fake features you wish to use.
            gen_labels = Variable((set_union_operation(labels1, labels2)).type(FloatTensor))
            # featureVecs = featureVecs.type(FloatTensor)
            # concat vector pairs and labels to create a batch
            if fakeI == 0:
                # concatenatedFeatureVecs1 = featureVec1.clone()
                concatenatedFeatureVecs1 = img1.clone()
                # concatenatedFeatureVecs2 = featureVec2.clone()
                concatenatedFeatureVecs2 = img2.clone()
                fakeLabels = gen_labels.clone()
            else:
                # concatenatedFeatureVecs1 = torch.cat((concatenatedFeatureVecs1, featureVec1.clone()), 0)
                concatenatedFeatureVecs1 = torch.cat((concatenatedFeatureVecs1, img1.clone()), 0)
                # concatenatedFeatureVecs2 = torch.cat((concatenatedFeatureVecs2, featureVec2.clone()), 0)
                concatenatedFeatureVecs2 = torch.cat((concatenatedFeatureVecs2, img2.clone()), 0)
                fakeLabels = torch.cat((fakeLabels, gen_labels.clone()), 0)

        concatenatedFeatureVecs1 = base_model(concatenatedFeatureVecs1)
        concatenatedFeatureVecs2 = base_model(concatenatedFeatureVecs2)

        concatenatedFeatureVecs1 = concatenatedFeatureVecs1.type(FloatTensor)
        concatenatedFeatureVecs2 = concatenatedFeatureVecs2.type(FloatTensor)
        fakeLabels = fakeLabels.type(FloatTensor)
        outputs_setopt = setops_model(concatenatedFeatureVecs1, concatenatedFeatureVecs2)
        _, _, a_S_b_em, _, a_U_b_em, _, a_I_b_em, _, _, _, _, _, _, _, _, _, _, _ = outputs_setopt
        # genFeatureVecs = generator(concatenatedFeatureVecs)
        # Use which ever fake vectors you want to use.
        genFeatureVecs = a_U_b_em
        # concat the fake vectors batch with the real vectors batch to create 1 unified batch
        inputs = torch.cat((realVecs, genFeatureVecs), 0)
        inputs = inputs.type(FloatTensor)
        targets = torch.cat((realLabels, fakeLabels), 0)
        targets = targets.type(FloatTensor)
        # ---------------------
        #  Train Classifier
        # ---------------------
        outputs = classifier(inputs)
        auxLoss = auxiliary_loss(outputs, targets)
        auxLoss.backward()
        optimizer_C.step()

        # precision calculations
        OutputsScores = F.sigmoid(outputs)
        OutputsScores_clone = OutputsScores.clone().data.cpu().numpy()
        concatenatedLabels_clone = targets.clone().data.cpu().numpy()
        # Concatenate all batches together so that we can check accuracy at the end of every epoch
        if i == 0:
            all_batch_outputs_scores = OutputsScores_clone
            all_batch_targets = concatenatedLabels_clone
        else:
            all_batch_outputs_scores = np.concatenate((all_batch_outputs_scores, OutputsScores_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, concatenatedLabels_clone), axis=0)
        # once in every 20 batches calculate and print accuracy for the current batch.
        if (batches_done % 20) != 0:
            continue
        OutputsScoresAux = OutputsScores.data.cpu().numpy()
        # calc intersection over union accuracy
        outputs_class_acc_IOU = IOU_real_vectors_accuracy((OutputsScoresAux >= 0.5), targets.data.cpu().numpy())
        # print progress and loss
        print("[Epoch %d/%d] [Batch %d/%d] [Classification loss: %f]" % (epoch, opt.n_epochs, i,
              len(train_loader), auxLoss.item()))
        logger.info("[Epoch {0}/{1}] [Batch {2}/{3}] [Classification loss: {4}]".format(epoch, opt.n_epochs, i,
                         len(train_loader), auxLoss.item()))
        # print classifier accuracy intersection over union
        print("IOU classifier accuracy {}".format(outputs_class_acc_IOU))
        logger.info("IOU classifier accuracy {}".format(outputs_class_acc_IOU))

    # Calculate and print mean average precision from precision recall graph
    average_precision_epoch = precision_recall_statistics(all_batch_outputs_scores, all_batch_targets)
    sum16 = 0
    for label in trainODDataset.classList:
        sum16 += average_precision_epoch[label]
    avg16 = sum16 / 16
    print('Training average precision score, macro-averaged over all 16 classes, epoch {}: {}'.format(epoch, avg16))
    logger.info(
        'Training average precision score, macro-averaged over all 16 classes, epoch {}: {}'.format(epoch, avg16))
    print("Classifier's learning rate: {}".format(get_learning_rate(optimizer_C)))
    logger.info("Classifier's learning rate: {}".format(get_learning_rate(optimizer_C)))

    # test every few epochs over the full validation dataset of real images only
    total_loss_class = 0
    if (epoch+1) % 5 == 0:
        print("in test")
        logger.info("in test")
        for i, (imgs, targets) in enumerate(test_loader):
            classifier.eval()
            batches_done = epoch * len(train_loader) + i
            batch_size = targets.shape[0]
            imgs = imgs.type(FloatTensor)
            inputs = base_model(imgs)
            inputs = inputs.type(FloatTensor)
            targets = targets.type(FloatTensor)
            # Generate a batch of images
            real_aux = classifier(inputs)
            class_real_aux_loss = auxiliary_loss(real_aux, targets)
            total_loss_class += class_real_aux_loss.item()
            # Calculate classification accuracy
            real_aux_sig = F.sigmoid(real_aux)
            real_aux_sig_clone = real_aux_sig.clone().data.cpu().numpy()
            concatenatedLabels_clone = targets.clone().data.cpu().numpy()
            # Concatenate all batches together so that we can check accuracy at the end of every epoch
            if i == 0:
                all_batch_real_outputs = real_aux_sig_clone
                all_batch_real_targets = concatenatedLabels_clone
            else:
                all_batch_real_outputs = np.concatenate((all_batch_real_outputs, real_aux_sig_clone), axis=0)
                all_batch_real_targets = np.concatenate((all_batch_real_targets, concatenatedLabels_clone), axis=0)
        average_loss_class = total_loss_class / len(test_loader)
        # Calculate and print mean average precision from precision recall graph
        average_precision_batch_real = precision_recall_statistics(all_batch_real_outputs, all_batch_real_targets)
        sum16 = 0
        for label in trainODDataset.classList:
            sum16 += average_precision_batch_real[label]
        avg16 = sum16 / 16
        print('Test real average precision score, macro-averaged over all 16 classes, epoch {}: {}'.format(epoch, avg16))
        logger.info('Test real average precision score, macro-averaged over all 16 classes, epoch {}: {}'.format(epoch, avg16))
        # Calculate and print real IoU accuracy
        classifier_test_real_accuracy_IOU = IOU_real_vectors_accuracy((all_batch_real_outputs >= 0.5),
                                                                      all_batch_real_targets)
        print("Test, real data classifier IOU accuracy {}".format(classifier_test_real_accuracy_IOU))
        logger.info("Test, real data classifier IOU accuracy {}".format(classifier_test_real_accuracy_IOU))
        classifier.train()
        # save model
        is_best = classifier_test_real_accuracy_IOU > best_test_acc
        best_test_acc = max(classifier_test_real_accuracy_IOU, best_test_acc)
        classifier_file_name = 'classifier' + str(vis_file_out)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': classifier.state_dict(),
                         'acc': classifier_test_real_accuracy_IOU, 'best_acc': best_test_acc,
                         'optimizer': optimizer_C.state_dict()},
                        is_best, epoch, checkpoint=opt.results_path, filename=classifier_file_name)


# save model
classifier_file_name = 'classifier' + str(vis_file_out) + 'final'
save_checkpoint({'epoch': epoch + 1, 'state_dict': classifier.state_dict(),
                 'acc': classifier_test_real_accuracy_IOU, 'best_acc': best_test_acc,
                 'optimizer': optimizer_C.state_dict()},
                is_best, epoch, checkpoint=opt.results_path, filename=classifier_file_name)


