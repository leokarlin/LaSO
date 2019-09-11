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

The code here includes a training script to train new LaSO networks and test scripts for precision, image retrieval and multi-label few shot classification.

Running the code 
==================
Setup
------------------
- Create a conda environment which will automatically install necessary packages.


 $ conda create --name myenv --file spec-file.txt

- Download the coco data, save it and point to it in the script or flags.
- Clone this git directory
- Download the pretrained models - https://drive.google.com/drive/folders/1FWg9gWM37SXk-bYOBH3-3Oe7uayzfiOd?usp=sharing
- Manually install needed packages -

 $ cd LaSO 

 $ python setup.py build develop 


 

Running the code
------------------
Train LaSO from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Leave variables which have a default value to their default value.
This code supports MLflow, you just need to set the environment variable MLFLOW_SERVER with the adress of your mlflow server.

We used a remote mlflow server, the server creation command is: $mlflow server --host 0.0.0.0 --file-store <BASE_STORAGE>/mlflow/mlruns 

more information can be find in the mlflow website: https://www.mlflow.org/docs/latest/tutorial.html

running training using resnet:

 $ cd scripts_coco

 $ python train_setops_stripped.py --inception_transform_input=False --resume_path=<path_to_LaSO_models>/resnet_base_model_only --resume_epoch=49 --init_inception=False --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=2 --base_network_name=resnet50 --crop_size=224 --epochs=50 --train_base=False --coco_path=<path to local coco folder> --results_path=<base path for results>

running training using inception paper model:

 $ cd scripts_coco

 $ python train_setops_stripped.py --inception_transform_input=False --resume_path=<path_to_LaSO_models>/paperBaseModel --init_inception=True --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=2 --base_network_name=Inception3 --crop_size=299 --epochs=50 --train_base=False --coco_path=<path to local coco folder> --results_path=<base path for results>

Reproduce the paper's results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test mAP precision do:

Leave variables which have a default value to their default value.

 $ cd scripts_coco

 $ python test_precision.py --unseen=False --resume_path=<path_to_LaSO_models> --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=Inception3 --init_inception=True --crop_size=299 --skip_tests=1 --paper_reproduce=True --coco_path=<path to local coco folder> --results_path=<base path for results>

Toggle unseen to True to test for unseen during training classes

To test retrieval do:

Leave variables which have a default value to their default value.

 $ cd scripts_coco

 $ python test_retrieval.py --unseen=False --resume_path=<path_to_LaSO_models> --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=Inception3 --init_inception=True --crop_size=299 --skip_tests=1 --paper_reproduce=True --metric=minkowski --tree_type=BallTree --coco_path=<path to local coco folder> --results_path=<base path for results>



Generate the Resnet model's results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test mAP precision do:

Leave variables which have a default value to their default value.

 $ cd scripts_coco

 $ python test_precision.py --unseen=False --resume_path=<path_to_LaSO_models>/resnet_LaSO_models/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10 --coco_path=<path to local coco folder> --results_path=<base path for results>

 $ python test_precision.py --unseen=True --resume_path=<path_to_LaSO_models>/resnet_LaSO_models/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10 --coco_path=<path to local coco folder> --results_path=<base path for results>

To test retrieval do:

Leave variables which have a default value to their default value.

 $ cd scripts_coco

 $ python test_retrieval.py --unseen=False --resume_path=<path_to_LaSO_models>/resnet_LaSO_models/ --sets_basic_block_name=SetopResBasicBlock --sets_block_name=SetopResBlock_v1 --sets_network_name=SetOpsResModule --ops_latent_dim=8092 --ops_layer_num=1 --resume_epoch=4 --base_network_name=resnet50 --init_inception=False --crop_size=299 --skip_tests=1 --avgpool_kernel=10 --metric=minkowski --tree_type=BallTree --coco_path=<path to local coco folder> --results_path=<base path for results>

Toggle unseen to True to test for unseen during training classes

Expected results
^^^^^^^^^^^^^^^^

.. image:: https://i.ibb.co/GkYdnM2/readme-results-table.png


Generate the augmentation model's results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paper model 1 shot -

 $ python test_augmentation.py --base_network_name='Inception3' --batch_size=4 --class_cap=1 --class_ind_dict_path='<LaSO folder>/data_for_augmentation/1shotRun1ClassIdxDict.pkl' --classifier_name='Inception3Classifier' --crop_size=299  --g_inner_dim=2048 --init_inception=1 --latent_dim=2048 --lr=0.01 --n_epochs=50 --paper_reproduce=1 --resume_path=<path_to_LaSO_models> --sets_basic_block_name='SetopResBasicBlock' --sets_block_name='SetopResBlock_v1' --sets_network_name='SetOpsResModule' --used_ind_path='<LaSO folder>/data_for_augmentation/1shotRun1UsedIndices.pkl' --results_path=<folder path to save models> --coco_path=<path to local coco folder>

Paper model 5 shot - 

 $ python test_augmentation.py --base_network_name='Inception3' --batch_size=4 --class_cap=5 --class_ind_dict_path='<LaSO folder>/data_for_augmentation/5shotRun1ClassIdxDict.pkl' --classifier_name='Inception3Classifier' --crop_size=299  --g_inner_dim=2048 --init_inception=1 --latent_dim=2048 --lr=0.01 --n_epochs=50 --paper_reproduce=1 --resume_path=<path_to_LaSO_models> --sets_basic_block_name='SetopResBasicBlock' --sets_block_name='SetopResBlock_v1' --sets_network_name='SetOpsResModule' --used_ind_path='<LaSO folder>/data_for_augmentation/5shotRun1UsedIndices.pkl' --results_path=<folder path to save models> --coco_path=<path to local coco folder>
