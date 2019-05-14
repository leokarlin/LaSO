import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import random
import torch.nn.functional as F

use_cuda = True if torch.cuda.is_available() else False
random.seed(5)
torch.manual_seed(5)
if use_cuda:
    torch.cuda.manual_seed_all(5)


class Img2OurVec():
    #def __init__(self, model='inception_v3', layer='default', layer_output_size=2048):
    def __init__(self, model='inception', layer='default', layer_output_size=2048, data="top10", transform=None):
        """ Img2Vec
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        cuda = True if torch.cuda.is_available() else False

        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        # self.model_path = '/dccstor/alfassy/saved_models/inception_traincocoInceptionT10Half2018.9.1.9:30epoch:71'
        # self.model_path = '/dccstor/alfassy/saved_models/inception_trainCocoIncHalf2018.10.3.13:39best'
        # self.model_path = '/dccstor/alfassy/saved_models/inception_trainCocoIncHalf2018.10.8.12:46best'
        self.model_path = '/dccstor/alfassy/saved_models/inception_trainCocoIncHalf642018.10.9.13:44epoch:30'
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer, data)
        self.model = self.model.to(self.device)
        self.model.eval()
        #self.scaler = transforms.Resize(224, 224)
        #self.scaler = transforms.Scale((224, 224))
        self.transform = transform
        self.model_name = model

    def get_vec(self, image, tensor=True):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0).to(self.device)

        batch_size = image.shape[0]

        # print(image.shape)
        if self.model_name == "inception":
            my_embedding = torch.zeros(batch_size, self.layer_output_size, 8, 8).to(self.device)

        else:
            my_embedding = torch.zeros(batch_size, self.layer_output_size, 1, 1).to(self.device)

        def copy_data_resnet(m, i, o):
            my_embedding.copy_(o.data)

        def copy_data_inception(m, i, o):
            my_embedding.copy_(i.data)

        if self.model_name == "inception":
            h = self.extraction_layer.register_forward_hook(copy_data_resnet)
        else:
            h = self.extraction_layer.register_forward_hook(copy_data_resnet)
        h_x = self.model(image)
        h.remove()
        # print(my_embedding.shape)
        my_embedding = F.avg_pool2d(my_embedding, kernel_size=8)

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer, data):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if data == "full":
            out_size = 200
        else:
            out_size = 80

        if model_name == 'inception':
            model = models.inception_v3(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_size)
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, out_size)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_size)
        elif model_name == 'ourT10Class':
            # model = torch.load('/dccstor/alfassy/saved_models/trained_discriminatorfeatureClassifierTrain2018.8.22.12:54epoch:128')
            model = torch.load('/dccstor/alfassy/saved_models/inception_trainincT10Half2018.9.4.14:40epoch:26')
        else:
            raise KeyError('Model %s was not found' % model_name)

        model.eval()

        if use_cuda:
            model.cuda()

        if model_name == 'inception' or model_name == 'resnet18':
            # Load checkpoint.
            assert os.path.isfile(self.model_path), 'Error: no checkpoint found!'
            checkpoint = torch.load(self.model_path)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        if model_name == 'inception':
            if layer == 'default':
                layer = model._modules.get('Mixed_7c')
                self.layer_output_size = 2048
            else:
                raise Exception('wrong layer name')
            return model, layer
        elif model_name == 'resnet18':
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                raise Exception('wrong layer name')
            return model, layer
        elif model_name == 'ourT10Class':
            layer = model._modules.get('linear_block')
            self.layer_output_size = 2048
            return model, layer