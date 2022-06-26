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
    VGG Model.

    Reference:
      Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015)
      [https://arxiv.org/abs/1409.1556]
"""

import flax.linen as nn
import jax
from jax import numpy as jnp
from functools import partial
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model
from typing import List


class VGGBlock(nn.Module):

    """
        VGG Block Class.

        Attributes:
            num_conv : the number of convolutional layers to use in the block.
            filters : the number of filters to use in one convolutional
              layer.
            use_bn : flags regarding whether to apply batch norm after
              convolution. 
    """

    num_conv : int
    filters : int
    use_bn : bool = True

    @nn.compact
    def __call__(self,
                x : jnp.ndarray,
                train : bool):
        """
            Apply VGGBlock module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block
        """
        for _ in range(self.num_conv):
            x = nn.Conv(self.filters, (3, 3), strides = (1, 1), padding = "SAME",
                    use_bias = True, kernel_init=conv_kernel_init)(x)
            if self.use_bn:
                x = ActivationOp()(x, train = train)
            else:
                x = jax.nn.relu(x)

        x = nn.max_pool(x, (2, 2), (2, 2))
        return x


class VGGBaseClass(nn.Module):

    """
        VGG Base Class.

        Attributes:
            block_sizes : the list collects how many convolutional layers would
              be applied in each block.
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer. 
            base_filter : the base number of filters to use in the blocks. The
              number of filters would increase by a factor of 2 ** block_index
              (maximum 512 filters). 
            use_bn : flags regarding whether to apply batch norm in VGGBlock.
    """

    block_sizes : List
    num_outputs: int
    base_filter : int = 64
    use_bn : bool = True
    fc_reg : str = "dropout"

    @nn.compact
    def __call__(self,
                inputs: jnp.ndarray,
                train: bool) -> jnp.ndarray:
        """
            Apply VGGBase module.

            Args:
                inputs : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not and whether to apply dropout.

            Returns:
                x : output of the module.
        """
        x = inputs
        for i, conv_num in enumerate(self.block_sizes):
            x = VGGBlock(conv_num, int(min(64 * (2 ** i), 512)),
                     use_bn = self.use_bn)(x, train = train)

        x = x.reshape((x.shape[0], -1)) 
        for _ in range(2):
            x = nn.Dense(4096, kernel_init=dense_layer_init_fn)(x)
            if self.fc_reg == "dropout":
                x = jax.nn.relu(x)
                x = nn.Dropout(0.5)(x, deterministic=not train)
            elif self.fc_reg == "bn":
                x = ActivationOp()(x, train = train)
            else:
                x = jax.nn.relu(x)

        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)

        return x


@_register_model("VGG16")
def VGG16BN(num_outputs : int, *args, **kwargs):
    """
        Build VGG16 module. block_list = [2, 2, 3, 3, 3]

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VGGBaseClass : a VGG16 Module.
    """
    return VGGBaseClass([2, 2, 3, 3, 3],
                        num_outputs = num_outputs,
                        use_bn = False,
                        *args, **kwargs)


@_register_model("VGG19")
def VGG19BN(num_outputs : int, *args, **kwargs):
    """
        Build VGG19 module. block_list = [2, 2, 4, 4, 4]

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VGGBaseClass : a VGG19 Module.
    """    
    return VGGBaseClass([2, 2, 4, 4, 4],
                        num_outputs = num_outputs,
                        use_bn = False,
                        *args, **kwargs)


@_register_model("VGG16BN")
def VGG16BN(num_outputs : int, *args, **kwargs):
    """
        Build VGG16BN module. block_list = [2, 2, 3, 3, 3]

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VGGBaseClass : a VGG16BN Module.
    """
    return VGGBaseClass([2, 2, 3, 3, 3],
                        num_outputs = num_outputs,
                        use_bn = True,
                        *args, **kwargs)

@_register_model("VGG19BN")
def VGG19BN(num_outputs : int, *args, **kwargs):
    """
        Build VGG19BN module. block_list = [2, 2, 4, 4, 4]

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VGGBaseClass : a VGG19BN Module.
    """    
    return VGGBaseClass([2, 2, 4, 4, 4],
                        num_outputs = num_outputs,
                        use_bn = True,
                        *args, **kwargs)
