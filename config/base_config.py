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

"""Base config flags, including basic config flags ."""

import ml_collections


def get_basic_config():

    config = ml_collections.ConfigDict()

    config.model_folder = None
    config.init_seeds = 0
    config.seeds = 0
    config.batch_size = 128 * 2
    config.base_lr = 1e-1
    config.total_epochs = 200
    config.gradient_clipping = 5.0
    config.use_learning_rate_schedule = True
    config.l2_regularization = 1e-3
    config.use_rmsprop = False
    config.lr_schedule_type = "cosine"
    config.save_ckpt_every_n_epochs = 200
    config.warmup_steps = 0
    config.label_smoothing = 0.

    config.additional_checkpoints_at_epochs = []
    config.also_eval_on_training_set = False
    config.compute_top_5_error_rate = False
    config.evaluate_every = 1
    config.inner_group_size = None
    config.no_weight_decay_on_bn = False
    config.asam = False
    config.use_dual_in_adam = False

    config.gnp = ml_collections.ConfigDict()
    config.gnp.r = 0.0
    config.gnp.sync_perturbations = False
    config.gnp.alpha = 0.0
    config.gnp.norm_perturbations = False

    config.retrain = False
    config.logging = ml_collections.ConfigDict()
    config.logging.tensorboard_logging_frequency = 1
    config.logging.basic_logger_level = "debug"
    config.logging.logger_sys_output = False
    config.write_config_to_json = True

    return config.lock()

def get_dataset_config():

    config = ml_collections.ConfigDict()
    config.dataset_name = "cifar10"
    config.image_level_augmentations = "basic"
    config.batch_level_augmentations = "none"
    config.image_size = 32
    config.num_classes = 10
    config.num_channels = 3

    # config.dataset_name = "imagenet"
    # config.image_level_augmentations = "none"
    # config.batch_level_augmentations = "none"
    # config.image_size = 224
    # config.num_classes = 1000
    # config.num_channels = 3

    return config


def get_optimizer_config():

    config = ml_collections.ConfigDict()

    config.opt_type = "SGD"

    config.opt_params = ml_collections.ConfigDict()
    config.opt_params.nesterov = True
    config.opt_params.beta = 0.9

    # config.opt_name = "Adam"
    # config.opt_params = ml_collections.ConfigDict()
    # config.opt_params.grad_norm_clip = 1.0
    # config.opt_params.weight_decay = 0.3

    return config
