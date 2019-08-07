"""Residual Arithmetic Operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import sys


class SetopResBasicBlock(nn.Module):
    """A basic setops residual layer.

    Applies Linear+BN on the input x and adds residual. Applies leaky-relu on top.
    """
    def __init__(self, latent_dim):
        super(SetopResBasicBlock, self).__init__()

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, residual):

        out = self.fc(x)
        out = self.bn(out)

        out += residual
        out = self.relu(out)

        return out


class SetopResBasicBlock_v1(nn.Module):
    """A gated setops residual layer.

    Applies Linear+BN on the input x and adds gated residual. Applies leaky-relu on top.
    """

    def __init__(self, latent_dim):
        super(SetopResBasicBlock_v1, self).__init__()

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.gate = nn.Linear(latent_dim, 1)

    def forward(self, x, residual):

        out = self.fc(x)
        out = self.bn(out)

        g = F.sigmoid(self.gate(x))
        out += g * residual

        out = self.relu(out)

        return out


class SetopResBlock(nn.Module):
    """Basic Set-Operation using residual.

    Args:
        input_dim:
        layers_num:
        arithm_op:

    Warining:
        You should probably not use this block as it does not have the RELU on top. The
        ReLU is needed for matching of the feature extractor.
    """

    def __init__(
            self,
            input_dim: int,
            layers_num: int,
            arithm_op: Callable,
            **kwargs):

        super(SetopResBlock, self).__init__()

        #
        # Build the network.
        #
        self.net_ab = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_layers = []
        for i in range(layers_num):
            layer_name = "res_layer{}".format(i)
            setattr(self, layer_name, SetopResBasicBlock(input_dim))
            self.res_layers.append(layer_name)

        self.fc_out = nn.Linear(input_dim, input_dim)
        self.arithm_op = arithm_op

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        out = torch.cat((a, b), dim=1)
        out = self.net_ab(out)

        res = self.arithm_op(a, b)

        for layer_name in self.res_layers:
            layer = getattr(self, layer_name)
            out = layer(out, res)

        out = self.fc_out(out)

        return out


class SetopResBlock_v1(nn.Module):
    """Basic Set-Operation using residual. Applies ReLU at the top (this is needed to match
     the embedding (output of the Inception before last layer).

    Args:
        input_dim:
        layers_num:
        arithm_op:
    """

    def __init__(
            self,
            input_dim: int,
            layers_num: int,
            arithm_op: Callable,
            basic_block_cls: nn.Module=SetopResBasicBlock,
            **kwargs):

        super(SetopResBlock_v1, self).__init__()

        #
        # Build the network.
        #
        self.net_ab = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_layers = []
        for i in range(layers_num):
            layer_name = "res_layer{}".format(i)
            setattr(self, layer_name, basic_block_cls(input_dim))
            self.res_layers.append(layer_name)

        self.fc_out = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)

        self.arithm_op = arithm_op

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        out = torch.cat((a, b), dim=1)
        out = self.net_ab(out)

        res = self.arithm_op(a, b)

        for layer_name in self.res_layers:
            layer = getattr(self, layer_name)
            out = layer(out, res)

        out = self.relu(self.fc_out(out))

        return out


class SetopResBlock_v2(nn.Module):
    """Basic Set-Operation using residual. Same as SetopResBlock_v1 but with dropout to avoid overfit
    of training data.

    Args:
        input_dim:
        layers_num:
        arithm_op:
    """

    def __init__(
            self,
            input_dim: int,
            layers_num: int,
            arithm_op: Callable,
            basic_block_cls: nn.Module=SetopResBasicBlock,
            dropout_ratio: float=0.5,
            **kwargs):

        super(SetopResBlock_v2, self).__init__()

        #
        # Build the network.
        #
        self.net_ab = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_layers = []
        for i in range(layers_num):
            layer_name = "res_layer{}".format(i)
            setattr(self, layer_name, basic_block_cls(input_dim))
            self.res_layers.append(layer_name)

        self.fc_out = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)

        self.arithm_op = arithm_op
        self.dropout_ratio = dropout_ratio

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        out = torch.cat((a, b), dim=1)
        out = self.net_ab(out)

        res = self.arithm_op(a, b)

        for layer_name in self.res_layers:
            layer = getattr(self, layer_name)
            out = layer(out, res)

        out = self.relu(self.fc_out(out))

        if self.dropout_ratio > 0:
            out = F.dropout(out, p=self.dropout_ratio, training=self.training)

        return out


def subrelu(x, y):
    return F.relu(x-y)


class SetOpsResModule(nn.Module):
    def __init__(
            self,
            input_dim,
            S_layers_num,
            I_layers_num,
            U_layers_num,
            block_cls_name="SetopResBlock",
            basic_block_cls_name="SetopResBasicBlock",
            dropout_ratio=0.5,
            **kwargs):

        super(SetOpsResModule, self).__init__()

        block_cls = getattr(sys.modules[__name__], block_cls_name)
        basic_block_cls = getattr(sys.modules[__name__], basic_block_cls_name)

        self.subtract_op = block_cls(
            input_dim=input_dim,
            layers_num=S_layers_num,
            arithm_op=subrelu,
            basic_block_cls=basic_block_cls,
            dropout_ratio=dropout_ratio,
            **kwargs
        )
        self.intersect_op = block_cls(
            input_dim=input_dim,
            layers_num=I_layers_num,
            arithm_op=torch.min,
            basic_block_cls=basic_block_cls,
            dropout_ratio=dropout_ratio,
            **kwargs
        )
        self.union_op = block_cls(
            input_dim=input_dim,
            layers_num=U_layers_num,
            arithm_op=torch.add,
            basic_block_cls=basic_block_cls,
            dropout_ratio=dropout_ratio,
            **kwargs
        )

    def forward(self, a, b):

        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)

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

        return out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
               a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
               a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a
