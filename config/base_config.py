# Copyright 2022 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Base config flags, including basic config flags, model config flags, dataset
      config flags and optimizer config flags.
"""

import ml_collections
from ml_collections import config_dict

def get_basic_config() -> ml_collections.ConfigDict:
    """
        Set basic config flags.

        Returns:
            config : the base config class.
    """

    config = ml_collections.ConfigDict()

    # model_folder : leave it None, and it will be set further.
    config.model_folder = None

    # init_seeds : the seeds for model initialization.
    config.init_seeds = 0

    # seeds : the seeds for setting PRNG keys during training.
    config.seeds = 0

    # batch_size : batch size. This should be divisible by your GPU numbers.
    config.batch_size = 128 * 2

    # base_lr : base learning rate. 
    config.base_lr = 1e-1

    # total_epochs : total epochs for training.
    config.total_epochs = 200

    # gradient_clipping : gradient clipping.
    config.gradient_clipping = 5.0

    # use_learning_rate_schedule : use_learning_rate_schedule
    config.use_learning_rate_schedule = True

    # l2_regularization : l2 regularization.
    config.l2_regularization = 1e-3

    # lr_schedule_type : learning rate schedule
    config.lr_schedule_type = "cosine"

    # save_ckpt_every_n_epochs : save model checkpoint every n epochs
    config.save_ckpt_every_n_epochs = 200

    # warmup_epochs : epochs for warmup training.
    config.warmup_epochs = 0

    # label_smoothing : label smooting rate.
    config.label_smoothing = 0.
    
    # compute_top_5_error_rate : record top 5 error rate
    config.compute_top_5_error_rate = False

    # evaluate_every_n_epochs : perform evaluation every n epoch.
    config.evaluate_every_n_epochs = 1

    # gnp params. See paper for details
    config.gnp = ml_collections.ConfigDict()
    config.gnp.r = 0.0
    config.gnp.alpha = 0.0
    config.gnp.norm_perturbations = True
    config.gnp.sync_perturbations = False

    config.gr_warmup_strategy = "none"

    # hybrid training flags.
    config.use_hybrid_training = False
    config.schedule_function = schedule_function
    config.hybrid_config = ml_collections.ConfigDict()
    config.hybrid_config.p = 0.

    # retain : if true, delete the exist model and train from the start.
    config.retrain = False

    # logging params.
    config.logging = ml_collections.ConfigDict()
    config.logging.tensorboard_logging_frequency = 1
    config.logging.basic_logger_level = "debug"
    config.logging.logger_sys_output = False
    config.write_config_to_json = True

    ### Some other flags
    config.use_test_set=True
    config.use_additional_skip_connections_in_wrn = False
    config.no_weight_decay_on_bn = False
    config.tfds_dir = "/raid/zy/datasets/tensorflow_datasets"
    config.imagenet_train_dir = "/raid/zy/datasets/imagenet/train"
    config.imagenet_val_dir = "/raid/zy/datasets/imagenet/val"

    return config.lock()


def get_dataset_config()-> ml_collections.ConfigDict:
    """
        Set dataset config flags. Datasets flags will be an attribute of the
          config class, i.e. config.dataset.dataset_flags.

        Returns:
            config : the dataset config class.
    """

    config = ml_collections.ConfigDict()
    config.dataset_name = "cifar10"
    config.image_level_augmentations = "basic"
    config.batch_level_augmentations = "none"
    config.image_size = 32
    config.num_classes = 10
    config.num_channels = 3

    return config


def get_optimizer_config() -> ml_collections.ConfigDict:
    """
        Set optimizer config flags. Optimizer flags will be an attribute of the
          config class, i.e. config.opt.opt_flags. Note that not all the flags
          will be pass to the optimizer class. These flags will further be
          filtered in the get_optimizer.py file based on the optimizer type.

        Returns:
            config : the optimizer config class. 
    """

    config = ml_collections.ConfigDict()
    config.opt_type = "SGD"
    config.opt_params = ml_collections.ConfigDict()
    config.opt_params.nesterov = True
    config.opt_params.momentum=0.9
    config.opt_params.grad_norm_clip = 5.0
    config.opt_params.weight_decay = 0.3

    return config


def get_model_config() -> ml_collections.ConfigDict:
    """
        Set model config flags. Model flags will be an attribute of the config
          class, i.e. config.model.model_flags. 

        Returns:
            config : the model config class.
    """

    config = ml_collections.ConfigDict()
    config.model_name = "WideResNet_28_10"
    
    # Config Other model parameters here. We will parse these parameters into
    # the model as key-value pair.
    config.model_params = ml_collections.ConfigDict()
    
    ### For example: ###
    # config.model_params.patches = ml_collections.ConfigDict({'size': (4, 4)})
    # config.model_params.hidden_size = 768
    # config.model_params.transformer = ml_collections.ConfigDict()
    # config.model_params.transformer.mlp_dim = 3072
    # config.model_params.transformer.num_heads = 12
    # config.model_params.transformer.num_layers = 12
    # config.model_params.transformer.attention_dropout_rate = 0.0
    # config.model_params.transformer.dropout_rate = 0.0
    
    return config

def schedule_function(p):
    def prob_func(step):
        return p
    return prob_func