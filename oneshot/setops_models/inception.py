"""Inception based base model and classifier for set-operations experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD
from torchvision.models.inception import InceptionE, InceptionAux, BasicConv2d


class Inception3(nn.Module):

    def __init__(self, num_classes=2048, aux_logits=True, transform_input=False, apply_avgpool=True):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.apply_avgpool = apply_avgpool

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048

        if self.apply_avgpool:
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            x = F.dropout(x, training=self.training)
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048

        if self.training and self.aux_logits:
            return x, aux

        return x


class Inception3Classifier(nn.Module):
    """Multi-Label classifier to apply above the inception feature extractor."""

    def __init__(self, num_classes=1000, **kwargs):
        super(Inception3Classifier, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        ## I am not sure why I had this code.
        # if self.training:
        #     x1, x2 = x
        #
        #     x1 = x1.view(x1.size(0), -1)
        #     x2 = x2.view(x2.size(0), -1)
        #
        #     return self.fc(x1), self.fc(x2)

        x = x.view(x.size(0), -1)
        return self.fc(x)


class Inception3SpatialAdapter(nn.Module):
    """Spatial avg pooling to apply between the inception features extractor
    or the setops networks and the multi-label classifier.
    """

    def __init__(self):
        super(Inception3SpatialAdapter, self).__init__()

    def forward(self, x):
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        return x


class SpatialConvolution(nn.Module):
    """Perform spatial convolution (without pooling) before the spatial setops network.

    Args:
        in_channels (int, optional) : Number of input channels.

    """
    def __init__(self, in_channels=2048):
        super(SpatialConvolution, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 1024, kernel_size=1)

        self.branch9x9_1 = BasicConv2d(in_channels, 512, kernel_size=1)
        self.branch9x9_2a = BasicConv2d(512, 512, kernel_size=(1, 9), padding=(0, 4))
        self.branch9x9_2b = BasicConv2d(512, 512, kernel_size=(9, 1), padding=(4, 0))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch9x9 = self.branch9x9_1(x)
        branch9x9 = [
            self.branch9x9_2a(branch9x9),
            self.branch9x9_2b(branch9x9),
        ]
        branch9x9 = torch.cat(branch9x9, 1)

        outputs = [branch1x1, branch9x9]
        return torch.cat(outputs, 1)


class SpatialConvolution_v1(nn.Module):
    """Perform (separated) spatial convolution (without pooling) before the spatial setops network.

    Args:
        in_channels (int, optional) : Number of input channels.
        spatial_kernel (int, optional) : Size of the spatial convolution kernel.
    """
    def __init__(self, in_channels=2048, spatial_kernel=9):
        super(SpatialConvolution_v1, self).__init__()

        out_channels1 = in_channels // 2
        out_channels2 = in_channels // 4

        self.branch1x1 = BasicConv2d(in_channels, out_channels1, kernel_size=1)
        self.branchNxN_1 = BasicConv2d(in_channels, out_channels2, kernel_size=1)

        spatial_padding = spatial_kernel // 2
        self.branchNxN_2a = BasicConv2d(out_channels2, out_channels2,
                                        kernel_size=(1, spatial_kernel), padding=(0, spatial_padding))
        self.branchNxN_2b = BasicConv2d(out_channels2, out_channels2,
                                        kernel_size=(spatial_kernel, 1), padding=(spatial_padding, 0))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branchNxN = self.branchNxN_1(x)
        branchNxN = [
            self.branchNxN_2a(branchNxN),
            self.branchNxN_2b(branchNxN),
        ]
        branchNxN = torch.cat(branchNxN, 1)

        outputs = [branch1x1, branchNxN]
        return torch.cat(outputs, 1)



class SetopsSpatialAdapter(nn.Module):
    """Adapter that applies the setops model on the spatial layer of the features extractor.

    The idea is to apply the setops network per "pixel" of the spatial output. Instead
    of for loop, the spatial layer is converted to channels.
    """

    def __init__(self, setops_model):
        super(SetopsSpatialAdapter, self).__init__()
        self.setops_model = setops_model

    def forward(self, a, b):
        batch_size, dim, m, n = a.shape
        a_ = a.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)
        b_ = b.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)

        outputs_ = self.setops_model(a_, b_)
        outputs = [output_.reshape(batch_size, m * n, dim).transpose(1, 2).view(batch_size, dim, m, n) for output_ in
                   outputs_]

        return outputs


class SetopsSpatialAdapter_v1(nn.Module):
    """Adapter that applies the setops model on the spatial layer of the features extractor.

    The idea is to apply the setops network per "pixel" of the spatial output. Instead
    of for loop, the spatial layer is converted to channels.

    In this version a spatial convolution is applied before the LaSO networks.
    """

    def __init__(self, setops_model):
        super(SetopsSpatialAdapter_v1, self).__init__()
        self.setops_model = setops_model
        self.spatial_convolution = SpatialConvolution()

    def forward(self, a, b):
        batch_size, dim, m, n = a.shape

        a = self.spatial_convolution(a)
        b = self.spatial_convolution(b)

        a_ = a.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)
        b_ = b.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)

        outputs_ = self.setops_model(a_, b_)
        outputs = [output_.reshape(batch_size, m * n, dim).transpose(1, 2).view(batch_size, dim, m, n) for output_ in
                   outputs_]

        return outputs


def inception3_ids(num_attributes, ids_embedding_size, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Inception3()
    classifier = Inception3Classifier(num_classes=num_attributes, **kwargs)
    classifier_ids = Inception3Classifier(num_classes=ids_embedding_size, **kwargs)
    if pretrained:
        raise NotImplemented("pretrained parameter not implemented.")
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model, classifier, classifier_ids


###############################################################################
###############################################################################
#
# Cutting the Inception3 network deeper.
#
class Inception3_6e(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, apply_avgpool=True):
        super(Inception3_6e, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.apply_avgpool = apply_avgpool

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        return x


class Inception3SpatialAdapter_6e(nn.Module):
    """Spatial avg pooling to apply between the inception features extractor
    or the setops networks and the multi-label classifier.
    """

    def __init__(self):
        super(Inception3SpatialAdapter_6e, self).__init__()

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        return x

class SetopsSpatialAdapter_6e(nn.Module):
    """Adapter that applies the setops model on the spatial layer of the features extractor.

    The idea is to apply the setops network per "pixel" of the spatial output. Instead
    of for loop, the spatial layer is converted to channels.

    In this version a spatial convolution is applied before the LaSO networks.
    """

    def __init__(self, setops_model):
        super(SetopsSpatialAdapter_6e, self).__init__()
        self.setops_model = setops_model
        self.spatial_convolution = SpatialConvolution_v1(in_channels=768, spatial_kernel=17)

    def forward(self, a, b):
        batch_size, dim, m, n = a.shape

        a = self.spatial_convolution(a)
        b = self.spatial_convolution(b)

        a_ = a.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)
        b_ = b.view(batch_size, dim, m * n).transpose(1, 2).reshape(batch_size * m * n, dim)

        outputs_ = self.setops_model(a_, b_)
        outputs = [output_.reshape(batch_size, m * n, dim).transpose(1, 2).view(batch_size, dim, m, n) for output_ in
                   outputs_]

        return outputs


