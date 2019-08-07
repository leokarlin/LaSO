"""Models of discriminators, i.e. classifiers of unseen classes.
"""

from torch import nn
from oneshot.coco import COCO_LABELS_NUM


class AmitDiscriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes=COCO_LABELS_NUM, dropout_ratio=0.5, **kwargs):

        super(AmitDiscriminator, self).__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #
        # Output layers
        #
        self.aux_layer = nn.Sequential(nn.Linear(latent_dim, n_classes))

    def forward(self, feature_vec):

        out = self.linear_block(feature_vec)
        label = self.aux_layer(out)
        return label


class Discriminator1Layer(nn.Module):

    def __init__(self, input_dim, n_classes=COCO_LABELS_NUM, **kwargs):
        super(Discriminator1Layer, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class Discriminator2Layer(nn.Module):

    def __init__(self, input_dim, latent_dim, n_classes=COCO_LABELS_NUM, dropout_ratio=0.5, **kwargs):

        super(Discriminator2Layer, self).__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_ratio),
        )

        #
        # Output layers
        #
        self.aux_layer = nn.Sequential(nn.Linear(latent_dim, n_classes))

    def forward(self, feature_vec):

        out = self.linear_block(feature_vec)
        label = self.aux_layer(out)

        return label
