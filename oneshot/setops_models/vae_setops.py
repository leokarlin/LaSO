"""Variational Auto Encoder set operations.

Taken from:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class VAE(nn.Module):
    def __init__(self, input_dim=2048, vae_dim=128):
        super(VAE, self).__init__()

        self.fc0 = nn.Linear(input_dim, 784)
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, vae_dim)
        self.fc22 = nn.Linear(400, vae_dim)
        self.fc3 = nn.Linear(vae_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.fc5 = nn.Linear(784, input_dim)

    def encode(self, x):
        h = F.relu(self.fc0(x))
        h = F.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return F.relu(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_loss, recon_x, x, mu, logvar):
    """Reconstruction + KL divergence losses summed over all elements and batch
    """

    loss = recon_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + KLD


def subrelu(x, y):
    return F.relu(x-y)


class SetOpsVAEModule(nn.Module):
    def __init__(
            self,
            input_dim: int,
            vae_dim: int,
            vae_cls_name: str="VAE",
            **kwargs):

        super(SetOpsVAEModule, self).__init__()

        vae_cls = getattr(sys.modules[__name__], vae_cls_name)

        self.vae = vae_cls(
            input_dim=input_dim,
            vae_dim=vae_dim,
            **kwargs
        )

        self.subtract_op = subrelu
        self.intersect_op = torch.min
        self.union_op = torch.add

    def forward(self, a, b):

        a, logvar_a = self.vae.encode(a)
        b, logvar_b = self.vae.encode(b)

        logvar = (logvar_a + logvar_b) / 2

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
                   a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a]

        outputs = [self.vae.decode(self.vae.reparameterize(o, logvar)) for o in outputs]

        return outputs, a, logvar_a, b, logvar_b
