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
    Logger initialization, creating working folder and writing config files.
"""

import os
import logging
from absl import flags
import sys
from tensorflow.io import gfile
import json

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def init_logger() -> None:
    """
        Logger initialization. 
    """
    config = FLAGS.config
    logging_level_list = {"debug" : logging.DEBUG,
                        "info"  : logging.INFO,
                        "warning" : logging.WARNING,
                        "error" : logging.ERROR,
                        "critical" : logging.CRITICAL}
    assert config.logging.basic_logger_level in logging_level_list.keys()
    current_logging_level = logging_level_list[config.logging.basic_logger_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(current_logging_level)

    if config.logging.logger_sys_output:
        handler_sysout = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler_sysout.setFormatter(formatter)
        root_logger.addHandler(handler_sysout)

    root_logger.debug(f"Initilize logger, logger level {config.logging.basic_logger_level}.")


def create_model_folder() -> str:
    """
        Creating working folder. 

        Returns:
            model_folder_name : the full working folder name.
    """
    config = FLAGS.config
    model_folder_name = os.path.join(f"{FLAGS.working_dir}",
                                f"{config.model.model_name}",
                                f"{config.dataset.dataset_name}",
                                f"{config.dataset.image_level_augmentations}_{config.dataset.batch_level_augmentations}",
                                f"lr_{config.base_lr}",
                                f"bs_{config.batch_size}",
                                f"l2reg_{config.l2_regularization}",
                                f"grad_clip_{config.gradient_clipping}",
                                f"opt_{config.opt.opt_type}",
                                f"warmup_epochs_{config.warmup_epochs}",
                                f"r_{config.gnp.r}",
                                f"alpha_{config.gnp.alpha}",
                                f"epoch_{config.total_epochs}",
                                f"seeds_{config.init_seeds}_{config.seeds}",)

    logger.info(f"Model folder at {model_folder_name}")

    if not gfile.exists(model_folder_name):
        logger.info(f"Creating model folder at {model_folder_name}")
        gfile.makedirs(model_folder_name)
    else:
        if config.retrain:
            logger.info(f"Retraining. Deleting and creating the model folder {model_folder_name}")
            gfile.rmtree(model_folder_name)
            gfile.makedirs(model_folder_name)

    write_config_to_json(model_folder_name)
    return model_folder_name


def write_config_to_json(model_folder_name : str) -> None:
    """
        Writing current configs to json file.

        Args:
            model_folder_name : the path to the full working folder. 
    """    
    config = FLAGS.config
    if config.write_config_to_json:
        logger.info(f"Writing json to the model foler ...")
        with open(os.path.join(model_folder_name, "config.json"), "w") as f:
            try:
                json.dump(config.ConfigDict.to_json_best_effort(), f)
            except:
                logger.info("Can not convert to json.")