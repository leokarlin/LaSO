***************************************************************************
LaSO: Label-Set Operations networks for multi-label few-shot learning
***************************************************************************
Overview
============
This repository contains the implementation of "`LaSO: Label-Set Operations networks for multi-label few-shot learning <https://arxiv.org/abs/1902.09811>`_
" by Alfassy et al. 
It was posted on arxiv in Feb 2019 and will be presented in CVPR 2019.

In this paper we have presented the label set manipulation concept and have demonstrated its utility for a new and challenging
task of the multi-label few-shot classification. Our results show
that label set manipulation holds a good potential for this and potentially other interesting applications, and we hope that this paper
will convince more researchers to look into this interesting problem.

The code here includes a training script to train new LaSO networks and test scripts for both precision and image retrieval.

Running the code
==================
Setup
------------------
- Create a conda environment which will automatically install necessary packages.

   $ conda create --name myenv --file spec-file.txt

- Download the coco data, save it and point to it in the script.
- Install the experiment package from: link
- clone this git directory

  $ cd LaSO

  $ python setup.py build develop

 

Running the code
------------------
- In order to run LaSO, training you will need a backbone, you can train your own backbone (code not provided) or just use one of the two base model which we have provided (Inception/ Resnet)
- 

Train LaSO from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Leave variables which have a default value to their default value.

$ cd scripts_coco

$ python train_setops_stripped.py --inception_transform_input=False --resume_path=/dccstor/faceid/results/train_coco_resnet/0198_968f3cd/1174695/190117_081837 --resume_epoch=49 --init_inception=False --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --base_network_name=resnet50 --crop_size=224 --epochs=5 --train_base=False

Reproduce the paper's results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test mAP precision do:

Leave variables which have a default value to their default value.

$ cd scripts_coco

$ python test_precision.py --unseen=False --resume_path=/dccstor/alfassy/finalLaSO/code_release/trainedModels --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=Inception3 --init_inception=True --crop_size=299 --skip_tests=1 --paper_reproduce=True

Toggle unseen to True to test for unseen during training classes

To test retrieval do:

Leave variables which have a default value to their default value.

$ cd scripts_coco

$ python test_retrieval.py --unseen=False --resume_path=/dccstor/alfassy/finalLaSO/code_release/trainedModels --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=Inception3 --init_inception=True --crop_size=299 --skip_tests=1 --paper_reproduce=True --metric=minkowski --tree_type=BallTree



Generate the Resnet model's results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test mAP precision do:

Leave variables which have a default value to their default value.

$ cd scripts_coco

$ python test_precision.py --unseen=False --resume_path=/dccstor/alfassy/finalLaSO/code_release/trainedModels/resnet_model/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10

$ python test_precision.py --unseen=True --resume_path=/dccstor/alfassy/finalLaSO/code_release/trainedModels/resnet_model/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10

To test retrieval do:

Leave variables which have a default value to their default value.

$ cd scripts_coco

$ python test_retrieval.py --unseen=False --resume_path=/dccstor/alfassy/finalLaSO/code_release/trainedModels/resnet_model/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10 --metric=minkowski --tree_type=BallTree

Toggle unseen to True to test for unseen during training classes

.. image:: https://ibb.co/1Ky07Xq

