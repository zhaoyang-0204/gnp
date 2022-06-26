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
    Train config flags, which collects all the config flags. The path to this
    file must be parsed when executing the training script. 
"""

import ml_collections
from gnp.config.base_config import get_basic_config, get_optimizer_config, get_dataset_config, get_model_config


def get_config() -> ml_collections.ConfigDict:
    """
        Train config flags, which collects all the config flags.

        Returns:
            config : the final config class for training. 
    """

    config = get_basic_config()
    config.unlock()

    config.model = get_model_config()
    config.dataset = get_dataset_config()
    config.opt = get_optimizer_config()

    return config.lock()