"""Basic blocks for set-operations networks (LaSO).
"""

from .ae_setops import SetOpsAEModule

from .discriminators import AmitDiscriminator
from .discriminators import Discriminator1Layer
from .discriminators import Discriminator2Layer

from .inception import Inception3
from .inception import Inception3Classifier
from .inception import Inception3SpatialAdapter
from .inception import Inception3SpatialAdapter_6e
from .inception import inception3_ids
from .inception import Inception3_6e
from .inception import SpatialConvolution
from .inception import SpatialConvolution_v1
from .inception import SetopsSpatialAdapter
from .inception import SetopsSpatialAdapter_v1
from .inception import SetopsSpatialAdapter_6e

from .res_setops import SetopResBasicBlock
from .res_setops import SetopResBasicBlock_v1
from .res_setops import SetopResBlock
from .res_setops import SetopResBlock_v1
from .res_setops import SetopResBlock_v2
from .res_setops import SetOpsResModule

from .resnet import ResNet
from .resnet import ResNetClassifier
from .resnet import resnet18
from .resnet import resnet18_ids
from .resnet import resnet18_ids_pre_v2
from .resnet import resnet18_ids_pre_v3
from .resnet import resnet34
from .resnet import resnet34_ids
from .resnet import resnet34_v2
from .resnet import resnet34_ids_pre
from .resnet import resnet34_ids_pre_v2
from .resnet import resnet34_ids_pre_v3
from .resnet import resnet50
from .resnet import resnet50_ids_pre_v3
from .resnet import resnet101
from .resnet import resnet152

from .setops import AttrsClassifier
from .setops import AttrsClassifier_v2
from .setops import CelebAAttrClassifier
from .setops import IDsEmbedding
from .setops import SetOpBlock
from .setops import SetOpBlock_v2
from .setops import SetOpBlock_v3
from .setops import SetOpBlock_v4
from .setops import SetOpBlock_v5
from .setops import PaperGenerator
from .setops import SetOpsModule
from .setops import SetOpsModule_v2
from .setops import SetOpsModule_v3
from .setops import SetOpsModule_v4
from .setops import SetOpsModule_v5
from .setops import SetOpsModule_v6
from .setops import SetOpsModulePaper
from .setops import TopLayer