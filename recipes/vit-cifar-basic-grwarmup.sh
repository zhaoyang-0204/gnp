#!/bin/bash

### Set you would like to use. If not set, all the gpus will be used by default.
# export CUDA_VISIBLE_DEVICES=0,1,2,3

ENV_PATH=your-path-to-your-python-venv
source $ENV_PATH

SRC_PATH=your-path-to-this-code-repo
### Could be ../../ when you run bash command at this folder.
cd $SRC_PATH

### The training config py file.
TRAIN_CONFIG_DIR=your-path-to-your-training-config-file.py
## Could be gnp/config/train_config.py

### The output dir where your would like to store your results.
WORKING_DIR=your-path-to-your-output-dir

MODEL_NAME=ViT_S16

INIT_SEEDS=0
SEEDS=0
BATCH_SIZE=256
BASE_LR=1e-3
TOTAL_EPOCH=300
### Will use the weight decay in Adam optimizer.
L2_REG=0

DATASET_NAME=cifar10
DATASET_IMAGE_LEVEL_AUG=basic
DATASET_BATCH_LEVEL_AUG=none

OPT_TYPE=Adam
OPT_WEIGHT_DECAY=0.3

GNP_R=0.1
GNP_ALPHA=1.0

WARM_UP=40
GR_WARMUP_STRATEGY=zero

python3 -m gnp.main.main --config=${TRAIN_CONFIG_DIR} \
                         --working_dir=${WORKING_DIR} \
                         --config.init_seeds=$INIT_SEEDS \
                         --config.seeds=$SEEDS \
                         --config.batch_size=$BATCH_SIZE \
                         --config.base_lr=$BASE_LR \
                         --config.warmup_epochs=$WARM_UP \
                         --config.gr_warmup_strategy=$GR_WARMUP_STRATEGY \
                         --config.l2_regularization=$L2_REG \
                         --config.total_epochs=$TOTAL_EPOCH \
                         --config.dataset.dataset_name=$DATASET_NAME \
                         --config.dataset.batch_level_augmentations=$DATASET_BATCH_LEVEL_AUG \
                         --config.dataset.image_level_augmentations=$DATASET_IMAGE_LEVEL_AUG \
                         --config.opt.opt_type=$OPT_TYPE \
                         --config.opt.opt_params.weight_decay = $OPT_WEIGHT_DECAY \
                         --config.gnp.alpha=$GNP_ALPHA \
                         --config.gnp.r=$GNP_R \
                         --config.model.model_name=$MODEL_NAME \

