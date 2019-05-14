"""Auto Encoder set operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class BasicLayer(nn.Module):
    """A basic linear++ layer..

    Applies Linear+BN+leaky-relu on the input.
    """
    def __init__(self, dim, **kwargs):
        super(BasicLayer, self).__init__()

        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class SetopEncoderDecoder(nn.Module):
    """Basic Set-Operation Encoder Decoder Module.

    Args:
        input_dim:
        layers_num:
        arithm_op:
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            output_dim: int,
            layers_num: int,
            dropout_ratio: float,
            **kwargs):

        super(SetopEncoderDecoder, self).__init__()

        self.in_net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        #
        # Build the network.
        #
        self.layers = []
        for i in range(layers_num):
            layer_name = "ae_layer{}".format(i)
            setattr(self, layer_name, BasicLayer(latent_dim, **kwargs))
            self.layers.append(layer_name)

        self.out_net = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.Dropout(dropout_ratio),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.in_net(x)

        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            out = layer(out)

        out = self.out_net(out)

        return out


def subrelu(x, y):
    return F.relu(x-y)


class SetOpsAEModule(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            encoder_dim: int,
            layers_num: int,
            encoder_cls_name: str="SetopEncoderDecoder",
            decoder_cls_name: str="SetopEncoderDecoder",
            dropout_ratio: float=0.5,
            **kwargs):

        super(SetOpsAEModule, self).__init__()

        encoder_cls = getattr(sys.modules[__name__], encoder_cls_name)
        decoder_cls = getattr(sys.modules[__name__], decoder_cls_name)

        self.encoder = encoder_cls(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=encoder_dim,
            layers_num=layers_num,
            dropout_ratio=dropout_ratio,
            **kwargs
        )
        self.decoder = decoder_cls(
            input_dim=encoder_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            layers_num=layers_num,
            dropout_ratio=dropout_ratio,
            **kwargs
        )

        self.subtract_op = subrelu
        self.intersect_op = torch.min
        self.union_op = torch.add

    def forward(self, a, b):

        a = self.encoder(a)
        b = self.encoder(b)

        a_S_b = self.subtract_op(a, b)
        b_S_a = self.subtract_op(b, a)

        a_S_b_b = self.subtract_op(a_S_b, b)
        b_S_a_a = self.subtract_op(b_S_a, a)

        a_I_b = self.intersect_op(a, b)
        b_I_a = self.intersect_op(b, a)

        a_S_b_I_a = self.subtract_op(a, b_I_a)
        b_S_a_I_b = self.subtract_op(b, a_I_b)
        a_S_a_I_b = self.subtract_op(a, a_I_b)
        b_S_b_I_a = self.subtract_op(b, b_I_a)

        a_I_b_b = self.intersect_op(a_I_b, b)
        b_I_a_a = self.intersect_op(b_I_a, a)

        a_U_b = self.union_op(a, b)
        b_U_a = self.union_op(b, a)

        a_U_b_b = self.union_op(a_U_b, b)
        b_U_a_a = self.union_op(b_U_a, a)

        out_a = self.union_op(a_S_b_I_a, a_I_b)
        out_b = self.union_op(b_S_a_I_b, b_I_a)

        outputs = [out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a,
                   a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a,
                   a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a, a, b]

        outputs = [self.decoder(o) for o in outputs]

        return outputs
