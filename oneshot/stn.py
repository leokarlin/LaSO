"""Spatial Transformer Network.

Example taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
"""

import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from .wideresnet_places import BasicBlock
#from .wideresnet_places import ResNet


class STNResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(STNResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)
        x = self.relu(x)

        return x


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class STN(nn.Module):
    def __init__(self, reset_fc_loc=True):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = STNResNet(
            BasicBlock,
            layers=[1, 1, 1, 1],
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32*7*7, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        if reset_fc_loc:
            logging.info("Resetting the fc_loc network")
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

    def forward(self, x):
        #
        # Calculate the transform
        #
        xs = self.localization(x)
        xs = xs.view(-1, 32*7*7)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())

        #
        # transform the input
        #
        x = F.grid_sample(x, grid)

        return x

