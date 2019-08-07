"""ResNet based base model and classifier for set-operations experiments.
"""
import math
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import torchvision.models as zoo
import torch.utils.model_zoo as model_zoo


class ResNet(nn.Module):

    def __init__(self, block, layers, avgpool_kernel=7, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(avgpool_kernel, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ResNetClassifier(nn.Module):

    def __init__(self, block, num_classes=1000, **kwargs):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    classifier = ResNetClassifier(BasicBlock, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier


def resnet18_ids(num_attributes, ids_embedding_size, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    classifier = ResNetClassifier(BasicBlock, num_classes=num_attributes, **kwargs)
    classifier_ids = ResNetClassifier(BasicBlock, num_classes=ids_embedding_size, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier, classifier_ids


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    classifier = ResNetClassifier(BasicBlock, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    classifier = ResNetClassifier(Bottleneck, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    classifier = ResNetClassifier(Bottleneck, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    classifier = ResNetClassifier(Bottleneck, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet152'])
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k in model.state_dict()}
        )

    return model, classifier


def resnet34_ids(num_attributes, ids_embedding_size, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    classifier = ResNetClassifier(BasicBlock, num_classes=num_attributes, **kwargs)
    classifier_ids = ResNetClassifier(BasicBlock, num_classes=ids_embedding_size, **kwargs)
    if pretrained:
        raise NotImplemented("pretrained parameter not implemented.")
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model, classifier, classifier_ids


def resnet34_v2(
        num_attributes: int=40,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        dropout_ratio: float=0,
        pretrained: bool=False,
        **kwargs):
    """Constructs a ResNet-34 model.

    """

    model = zoo.resnet34(pretrained=pretrained)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-1])

    input_dim = 512 * BasicBlock.expansion

    classifier_attr = AttrsClassifier_v2(
        input_dim=512 * BasicBlock.expansion,
        num_attributes=num_attributes,
        latent_dim=latent_dim,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        **kwargs)

    return model, classifier_attr



def resnet34_ids_pre(num_attributes, ids_embedding_size, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = zoo.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    # Remove the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default

    model = nn.Sequential(*list(model.children())[:-1])

    classifier_attr = ResNetClassifier(BasicBlock, num_classes=num_attributes, **kwargs)
    classifier_ids = ResNetClassifier(BasicBlock, num_classes=ids_embedding_size, **kwargs)

    return model, classifier_attr, classifier_ids


def resnet34_ids_pre_v2(
        num_attributes: int,
        ids_embedding_size: int,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        ids_layer_num: int=0,
        dropout_ratio: float=0,
        pretrained: bool=True,
        **kwargs):
    """Constructs a ResNet-34 model.

    """

    model = zoo.resnet34(pretrained=True)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-1])

    input_dim = 512 * BasicBlock.expansion

    classifier_attr = TopLayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=num_attributes,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        **kwargs)

    embedding_ids = TopLayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=ids_embedding_size,
        layers_num=ids_layer_num,
        dropout_ratio=dropout_ratio,
        ** kwargs)

    return model, classifier_attr, embedding_ids


def resnet18_ids_pre_v2(
        num_attributes: int,
        ids_embedding_size: int,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        ids_layer_num: int=0,
        dropout_ratio: float=0,
        **kwargs):
    """Constructs a ResNet-18 model.

    """

    model = zoo.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-1])

    input_dim = 512 * BasicBlock.expansion

    classifier_attr = TopLayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=num_attributes,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        **kwargs)

    embedding_ids = TopLayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=ids_embedding_size,
        layers_num=ids_layer_num,
        dropout_ratio=dropout_ratio,
        ** kwargs)

    return model, classifier_attr, embedding_ids


def resnet18_ids_pre_v3(
        num_attributes: int=40,
        ids_embedding_size: int=512,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        ids_layer_num: int=0,
        dropout_ratio: float=0,
        pretrained: bool=False,
        **kwargs):
    """Constructs a ResNet-18 model.

    """

    model = zoo.resnet18(pretrained=pretrained)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-2])

    input_dim = 512 * BasicBlock.expansion

    classifier_attr = AttrsClassifier(
        input_dim=512 * BasicBlock.expansion,
        num_attributes=num_attributes,
        latent_dim=latent_dim,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        apply_spatial=True,
        **kwargs)

    embedding_ids = IDsEmbedding(
        input_dim=512 * BasicBlock.expansion,
        embedding_size=ids_embedding_size,
        latent_dim=latent_dim,
        layers_num=ids_layer_num,
        dropout_ratio=dropout_ratio,
        apply_avgpool=True,
        ** kwargs)

    return model, classifier_attr, embedding_ids


def resnet34_ids_pre_v3(
        num_attributes: int=40,
        ids_embedding_size: int=512,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        ids_layer_num: int=0,
        dropout_ratio: float=0,
        pretrained: bool=False,
        **kwargs):
    """Constructs a ResNet-34 model.

    """

    model = zoo.resnet34(pretrained=pretrained)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-2])

    input_dim = 512 * BasicBlock.expansion

    classifier_attr = AttrsClassifier(
        input_dim=512 * BasicBlock.expansion,
        num_attributes=num_attributes,
        latent_dim=latent_dim,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        apply_spatial=True,
        **kwargs)

    embedding_ids = IDsEmbedding(
        input_dim=512 * BasicBlock.expansion,
        embedding_size=ids_embedding_size,
        latent_dim=latent_dim,
        layers_num=ids_layer_num,
        dropout_ratio=dropout_ratio,
        apply_avgpool=True,
        ** kwargs)

    return model, classifier_attr, embedding_ids


def resnet50_ids_pre_v3(
        num_attributes: int=40,
        ids_embedding_size: int=512,
        latent_dim: int=1024,
        attr_layer_num: int=0,
        ids_layer_num: int=0,
        dropout_ratio: float=0,
        pretrained: bool=False,
        **kwargs):
    """Constructs a ResNet-34 model.

    """

    model = zoo.resnet50(pretrained=pretrained)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-2])

    input_dim = 512 * Bottleneck.expansion

    classifier_attr = AttrsClassifier(
        input_dim=512 * Bottleneck.expansion,
        num_attributes=num_attributes,
        latent_dim=latent_dim,
        layers_num=attr_layer_num,
        dropout_ratio=dropout_ratio,
        apply_spatial=True,
        **kwargs)

    embedding_ids = IDsEmbedding(
        input_dim=512 * Bottleneck.expansion,
        embedding_size=ids_embedding_size,
        latent_dim=latent_dim,
        layers_num=ids_layer_num,
        dropout_ratio=dropout_ratio,
        apply_avgpool=True,
        ** kwargs)

    return model, classifier_attr, embedding_ids