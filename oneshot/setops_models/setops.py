"""Basic building blocks for set-ops networks (LaSO).
"""
import torch
import torch.nn as nn


class SetOpBlock(nn.Module):
    """Basic block of a Set Operation.

    Note:
        Due to a typo, I actually never used the input_layer,
        so to be compatible I commented it.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(self, input_dim: int, latent_dim: int, layers_num: int=3, dropout_ratio: float=0.5):

        raise Exception("This module is buggy. It uses the same block object (shares weights between layers).")

        super(SetOpBlock, self).__init__()

        self.layers_num = layers_num
        #self.input_layer = nn.Linear(2*input_dim, latent_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.interm_block = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(p=dropout_ratio),
        )

        self.output_layer = nn.Linear(latent_dim, input_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        ab = torch.cat((a, b), dim=1)

        for i in range(self.layers_num):
            ab = self.interm_block(ab)

        out = self.output_layer(ab)

        return self.relu(out)


class SetOpBlock_v2(nn.Module):
    """Basic block of a Set Operation.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(self, input_dim: int, latent_dim: int, layers_num: int=3, dropout_ratio: float=0.5):

        super(SetOpBlock_v2, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        #
        # Build the network.
        #
        layers = [
            nn.Linear(2 * input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(self.latent_dim),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, input_dim),
                #nn.LeakyReLU(0.2, inplace=True)
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        ab = torch.cat((a, b), dim=1)
        out = self.net(ab)

        return out


class SetOpBlock_v3(nn.Module):
    """Basic block of a Set Operation.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            layers_num: int=3,
            dropout_ratio: float=0.5,
            apply_spatial: bool = False):

        super(SetOpBlock_v3, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        #
        # Build the network.
        #
        output_dim = input_dim
        self.apply_spatial = apply_spatial
        if self.apply_spatial:
            self.spatial = \
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        input_dim,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    nn.BatchNorm2d(input_dim)
            )
            output_dim = output_dim * 7 * 7

        layers = [
            nn.Linear(2 * input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(self.latent_dim),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, output_dim),
                #nn.LeakyReLU(0.2, inplace=True)
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        if self.apply_spatial:
            a = self.spatial(a)
            b = self.spatial(b)

            a = a.view(a.size(0), -1)
            b = b.view(b.size(0), -1)

        ab = torch.cat((a, b), dim=1)
        out = self.net(ab)

        if self.apply_spatial:
            out = out.view(out.size(0), -1, 7, 7)

        return out


class SetOpBlock_v4(nn.Module):
    """Basic block of a Set Operation.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            layers_num: int=3,
            dropout_ratio: float=0.5):

        super(SetOpBlock_v4, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        #
        # Build the network.
        #
        output_dim = input_dim
        layers = [
            nn.Linear(2 * input_dim, latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.BatchNorm1d(self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, output_dim),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        ab = torch.cat((a, b), dim=1).view(a.size(0), -1)
        out = self.net(ab)

        return out


class SetOpBlock_v5(nn.Module):
    """Basic block of a Set Operation.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            layers_num: int=3,
            dropout_ratio: float=0.5,
            **kwargs):

        super(SetOpBlock_v5, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        #
        # Build the network.
        #
        output_dim = input_dim
        layers = [
            nn.Linear(2 * input_dim, latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.BatchNorm1d(self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, output_dim),
                nn.ReLU(inplace=True)
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        ab = torch.cat((a, b), dim=1).view(a.size(0), -1)
        out = self.net(ab)

        return out


class PaperGenerator(nn.Module):
    def __init__(self, inner_dim):
        super(PaperGenerator, self).__init__()
        latent_dim = 2048
        self.l1 = nn.Sequential(nn.Linear(latent_dim*2, inner_dim))

        self.linear_block = nn.Sequential(
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(inner_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, gen_input):
        out = self.l1(gen_input)
        genFeatureVec = self.linear_block(out)
        return genFeatureVec


class SetOpsModule(nn.Module):
    def __init__(
            self,
            input_dim: int,
            S_latent_dim: int, S_layers_num: int,
            I_latent_dim: int, I_layers_num: int,
            U_latent_dim: int, U_layers_num: int,
            block_cls: nn.Module=SetOpBlock_v2,
            **kwds):

        super(SetOpsModule, self).__init__()

        self.subtract_op = block_cls(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            **kwds
        )
        self.intersect_op = block_cls(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            **kwds
        )
        self.union_op = block_cls(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            **kwds
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        a_S_b = self.subtract_op(a, b)
        b_S_a = self.subtract_op(b, a)

        a_I_b = self.intersect_op(a, b)
        b_I_a = self.intersect_op(b, a)

        a_U_b = self.union_op(a, b)
        b_U_a = self.union_op(b, a)

        out_a = self.union_op(a_S_b, a_I_b)
        out_b = self.union_op(b_S_a, b_I_a)

        return out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a


class SetOpsModule_v2(nn.Module):
    def __init__(
            self,
            input_dim: int,
            S_latent_dim: int, S_layers_num: int,
            I_latent_dim: int, I_layers_num: int,
            U_latent_dim: int, U_layers_num: int,
            block_cls: nn.Module=SetOpBlock_v2,
            **kwds):

        super(SetOpsModule_v2, self).__init__()

        self.subtract_op = block_cls(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            **kwds
        )
        self.intersect_op = block_cls(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            **kwds
        )
        self.union_op = block_cls(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            **kwds
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        a_S_b = self.subtract_op(a, b)
        b_S_a = self.subtract_op(b, a)

        a_S_b_b = self.subtract_op(a_S_b, b)
        b_S_a_a = self.subtract_op(b_S_a, a)

        a_I_b = self.intersect_op(a, b)
        b_I_a = self.intersect_op(b, a)

        a_I_b_b = self.intersect_op(a_I_b, b)
        b_I_a_a = self.intersect_op(b_I_a, a)

        a_U_b = self.union_op(a, b)
        b_U_a = self.union_op(b, a)

        a_U_b_b = self.union_op(a_U_b, b)
        b_U_a_a = self.union_op(b_U_a, a)

        out_a = self.union_op(a_S_b, a_I_b)
        out_b = self.union_op(b_S_a, b_I_a)

        return out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
               a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a


class SetOpsModule_v3(nn.Module):
    def __init__(
            self,
            input_dim,
            S_latent_dim, S_layers_num,
            I_latent_dim, I_layers_num,
            U_latent_dim, U_layers_num,
            **kwds):

        super(SetOpsModule_v3, self).__init__()

        self.subtract_op = SetOpBlock_v2(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            **kwds
        )
        self.intersect_op = SetOpBlock_v2(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            **kwds
        )
        self.union_op = SetOpBlock_v2(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            **kwds
        )

    def forward(self, a, b):

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


class SetOpsModule_v4(nn.Module):
    def __init__(
            self,
            input_dim,
            S_latent_dim, S_layers_num,
            I_latent_dim, I_layers_num,
            U_latent_dim, U_layers_num,
            apply_spatial=True,
            **kwds):

        super(SetOpsModule_v4, self).__init__()

        self.subtract_op = SetOpBlock_v3(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            apply_spatial=apply_spatial,
            **kwds
        )
        self.intersect_op = SetOpBlock_v3(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            apply_spatial=apply_spatial,
            **kwds
        )
        self.union_op = SetOpBlock_v3(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            apply_spatial=apply_spatial,
            **kwds
        )

    def forward(self, a, b):

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


class SetOpsModule_v5(nn.Module):
    def __init__(
            self,
            input_dim,
            S_latent_dim, S_layers_num,
            I_latent_dim, I_layers_num,
            U_latent_dim, U_layers_num,
            **kwds):

        super(SetOpsModule_v5, self).__init__()

        self.subtract_op = SetOpBlock_v4(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            **kwds
        )
        self.intersect_op = SetOpBlock_v4(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            **kwds
        )
        self.union_op = SetOpBlock_v4(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            **kwds
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


class SetOpsModule_v6(SetOpsModule_v5):
    def __init__(
            self,
            input_dim,
            S_latent_dim, S_layers_num,
            I_latent_dim, I_layers_num,
            U_latent_dim, U_layers_num,
            **kwds):

        super(SetOpsModule_v6, self).__init__(
            input_dim,
            S_latent_dim, S_layers_num,
            I_latent_dim, I_layers_num,
            U_latent_dim, U_layers_num,
            **kwds
        )

        self.subtract_op = SetOpBlock_v5(
            input_dim=input_dim,
            latent_dim=S_latent_dim,
            layers_num=S_layers_num,
            **kwds
        )
        self.intersect_op = SetOpBlock_v5(
            input_dim=input_dim,
            latent_dim=I_latent_dim,
            layers_num=I_layers_num,
            **kwds
        )
        self.union_op = SetOpBlock_v5(
            input_dim=input_dim,
            latent_dim=U_latent_dim,
            layers_num=U_layers_num,
            **kwds
        )


class SetOpsModulePaper(nn.Module):
    def __init__(self, models_path):
        super(SetOpsModulePaper, self).__init__()

        self.subtract_op = PaperGenerator(8192)
        self.intersect_op =PaperGenerator(2048)
        self.union_op = PaperGenerator(2048)

        checkpoint = torch.load(models_path / 'paperSubModel')
        self.subtract_op.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(models_path / 'paperInterModel')
        self.intersect_op.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(models_path / 'paperUnionModel')
        self.union_op.load_state_dict(checkpoint['state_dict'])

    def forward(self, a, b):

        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        a_b = torch.cat((a, b), 1)
        b_a = torch.cat((b, a), 1)

        a_S_b = self.subtract_op(a_b)
        b_S_a = self.subtract_op(b_a)

        a_S_b_b = None
        b_S_a_a = None

        a_I_b = self.intersect_op(a_b)
        b_I_a = self.intersect_op(b_a)

        a_S_b_I_a = None
        b_S_a_I_b = None
        a_S_a_I_b = None
        b_S_b_I_a = None

        a_I_b_b = None
        b_I_a_a = None

        a_U_b = self.union_op(a_b)
        b_U_a = self.union_op(b_a)

        a_U_b_b = None
        b_U_a_a = None

        out_a = None
        out_b = None

        return out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
               a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
               a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a


class IDsEmbedding(nn.Module):

    def __init__(
            self,
            input_dim: int,
            embedding_size: int=512,
            latent_dim: int=1024,
            layers_num: int=0,
            dropout_ratio: float=0,
            apply_avgpool: bool=False):

        super(IDsEmbedding, self).__init__()

        self.apply_avgpool = apply_avgpool
        if self.apply_avgpool:
            self.avgpool = nn.AvgPool2d(7, stride=1)

        #
        # Build the network.
        #
        layers = [
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, embedding_size),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.apply_avgpool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.net(x)


class AttrsClassifier(nn.Module):

    def __init__(
            self,
            input_dim: int,
            num_attributes: int=40,
            latent_dim: int=1024,
            layers_num: int=0,
            dropout_ratio: float=0,
            apply_spatial: bool=False):

        super(AttrsClassifier, self).__init__()

        self.apply_spatial = apply_spatial
        if self.apply_spatial:
            self.spatial = \
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        input_dim,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    nn.BatchNorm2d(input_dim)
            )

        #
        # Build the network.
        #
        layers = [
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, num_attributes),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.apply_spatial:
            x = self.spatial(x)

        x = x.view(x.size(0), -1)
        return self.net(x)


class AttrsClassifier_v2(nn.Module):

    def __init__(
            self,
            input_dim: int,
            num_attributes: int=40,
            latent_dim: int=1024,
            layers_num: int=0,
            dropout_ratio: float=0):

        super(AttrsClassifier_v2, self).__init__()

        #
        # Build the network.
        #
        layers = [
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(layers_num):
            layers.extend(
                [
                    nn.Linear(latent_dim, latent_dim),
                    nn.BatchNorm1d(latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, num_attributes),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class TopLayer(nn.Module):
    """A top layer to add above another base_model.

    Args:
        input_dim:
        latent_dim:
        layers_num:
        dropout_ratio:
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            output_dim: int,
            layers_num: int=0,
            dropout_ratio: float=0):

        super(TopLayer, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        #
        # Build the network.
        #
        layers = [
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(self.latent_dim),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))

        layers.extend(
            [
                nn.Linear(latent_dim, output_dim),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        x = x.view(x.size(0), -1)
        out = self.net(x)

        return out


class CelebAAttrClassifier(nn.Module):
    def __init__(self, input_dim: int, attributes_num: int, latent_dim: int, layers_num: int=1, apply_bn: bool=True):

        super(CelebAAttrClassifier, self).__init__()

        self.layers_num = layers_num
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if apply_bn:
            self.interm_block = nn.Sequential(
                nn.BatchNorm1d(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.interm_block = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.output_layer = nn.Linear(latent_dim, attributes_num)

    def forward(self, x: torch.Tensor):

        x = self.relu(self.input_layer(x))
        for i in range(self.layers_num):
            x = self.interm_block(x)
        out = self.relu(self.output_layer(x))

        return out




