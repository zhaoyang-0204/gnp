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
    Manage the optimizer config flags and get the optimizer instance.
"""

from absl import flags
from flax import optim
import flax
from gnp.optimizer.Adam_class import AdamOptimizer
from gnp.optimizer.SGD_class import SGDOptimizer

FLAGS = flags.FLAGS


def manage_optimizer_config() -> None:
    """
        Manage the optimizer config flags to fit the argument of 
        the selected optimizer.
    """

    FLAGS.config.unlock()
    opt_flags = flags.FLAGS.config.opt
    _AVAILABLE_FLAGS = dict(
        SGD = ("beta", "grad_norm_clip", "weight_decay", "nesterov"),
        Momentum = ("beta", "weight_decay", "nesterov"),
        Adam = ("beta1", "beta2", "eps", "weight_decay", "grad_norm_clip")
    )
    assert opt_flags.opt_type in _AVAILABLE_FLAGS.keys()

    # Remove unmatched optimizer flags.
    for opt_para_item in opt_flags.opt_params:
        if opt_para_item not in _AVAILABLE_FLAGS[opt_flags["opt_type"]]:
           delattr(opt_flags.opt_params, opt_para_item) 
    
    if opt_flags.opt_type in ("SGD", "Momentum"):
        # We set weight decay in SGD optimizer to 0, since it will be
        # re-implemented during training.
        opt_flags.opt_params.weight_decay = 0.0

    flags.FLAGS.config.lock()


def get_optimizer(params : flax.core.frozen_dict.FrozenDict,
                 learning_rate: float,
                 ) -> flax.optim.Optimizer :
    """
        Get optimizer instance. If you would like to define your custom
          optimizer, you could write an optimizer py file in this folder and
          then launch it here.

        Args:
            params : the parameters of the model. It is generally generate 
              after using the model.init().
            learning_rate : the learning rate parsed to optimizer instance.
              This actually will not affect the learning rate during training, 
              since we will re-compute the learning rate at each step.
        Returns:
            optimizer : the optimizer instance. 
    """

    if FLAGS.config.opt.opt_type == "SGD":
        optimizer_def = SGDOptimizer(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)
    elif FLAGS.config.opt.opt_type == "Momentum":
        optimizer_def = optim.Momentum(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)
    elif FLAGS.config.opt.opt_type == "Adam":
        optimizer_def = AdamOptimizer(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)
    else:
        raise ValueError("Unkown optimizer type, {FLAGS.config.opt.opt_type} !")

    optimizer = optimizer_def.create(params)
    return optimizer