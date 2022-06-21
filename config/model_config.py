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

import ml_collections


def get_model_config():

    config = ml_collections.ConfigDict()

    config.model_name = "WideResNet_28_10"
    # config.model_name = "VIT"
    
    # Config Model Parameters Here
    config.model_params = ml_collections.ConfigDict()
    
    # config.model_params.patches = ml_collections.ConfigDict({'size': (4, 4)})
    # config.model_params.hidden_size = 768
    # config.model_params.transformer = ml_collections.ConfigDict()
    # config.model_params.transformer.mlp_dim = 3072
    # config.model_params.transformer.num_heads = 12
    # config.model_params.transformer.num_layers = 12
    # config.model_params.transformer.attention_dropout_rate = 0.0
    # config.model_params.transformer.dropout_rate = 0.0
    
    return config