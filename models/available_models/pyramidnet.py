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
    PyramidNet Model. 

    Reference:
      Deep Pyramidal Residual Networks.
      [https://arxiv.org/pdf/1610.02915.pdf]

    From the reference, there are two kind of structures. Here, we implement the
      PyramidNet Add Structure. The difference between the two structures lies
      on the growing rules regarding the channels. For Add structure, channels
      are growth in an addition manner while they are increased in a exponential
      manner in PyramidNet Exp. If you would like to try the PyramidNet Exp, you
      could change the growing rule to exponential in the PyramidNetAdd Class.
"""


import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model


def _shortcut(x : jnp.ndarray,
              chn_out : int,
              strides : Tuple[int, int]):
    """
        ShortCut Class for PyramidNet. Pad the gap between the channel_in and
          channel_out to zero.

        Args:
            x : inputs.
            chn_out : the channel of the expected output.
            strides : the strides of polling for output.
    """

    chn_in = x.shape[3]
    if strides != (1, 1):
        x = nn.avg_pool(x, strides, strides)
    if chn_out != chn_in:
        diff = chn_out - chn_in
        x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [0, diff]])
    return x


class BottleneckBlock(nn.Module):
    """
        Bottlenet Block Class.

        Attributes:
            channels :the number of channels to use in one convolutional
              layer.
            strides : the strides of convlution.
    """

    channels : int
    strides : int

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply Bottlenet Block module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block.
        """
        
        residual = x
        x = ActivationOp()(x, train = train, apply_relu=False)
        x = nn.Conv(self.channels, (1, 1), padding = "SAME",
                     use_bias = False, kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train)

        x = nn.Conv(self.channels, (3, 3), strides = self.strides, padding = "SAME",
                     use_bias = False, kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train)

        x = nn.Conv(self.channels * 4, (1, 1), padding = "SAME",
                     use_bias = False, kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train, apply_relu=False)

        residual = _shortcut(residual, self.channels * 4, self.strides)
        return x + residual


@_register_model("PyramidNetBase")
class PyramidNetAdd(nn.Module):
    """
        PyramidNet Add module.

        Attributes:
            pyramid_alpha : the increase factor of channels, whihc is 
              D_k+1 = D_k + alpha/total_blocks.
            pyramid_depth : the total depth of the convolutional layers. For
              bottleneck structure, the depth - 2 must be divisible by 9.
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
    """

    pyramid_alpha : int
    pyramid_depth : int
    num_outputs : int

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool):
        """
            Apply PyramidNet Add module.

            Args:
                inputs : inputs to the module.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : outputs of the module.
        """

        x = inputs
        assert (self.pyramid_depth - 2) % 9 == 0 # Bottlenet Arch
        blocks_per_group = (self.pyramid_depth - 2) // 9 # //6 if basic block
        num_channels = 16
        total_blocks = blocks_per_group * 3
        delta_channels = self.pyramid_alpha / total_blocks

        x = nn.Conv(16, (3, 3), padding = "SAME",
                use_bias = False, kernel_init=conv_kernel_init)(x)

        layer_num = 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            x = BottleneckBlock(channels = int(round(num_channels)), strides = (1, 1))(x, train = train)
            layer_num += 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            x = BottleneckBlock(channels = int(round(num_channels)),
                         strides = (2, 2) if block_i == 0 else (1, 1))(x, train = train)
            layer_num += 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            x = BottleneckBlock(channels = int(round(num_channels)),
                         strides = (2, 2) if block_i == 0 else (1, 1))(x, train = train)
            layer_num += 1

        assert layer_num - 1 == total_blocks
        x = ActivationOp()(x, train = train)
        x = nn.avg_pool(x, (8, 8))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
        return x


@_register_model("PyramidNet_164_48")
def PyramidNet_164_48(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-164-48 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetAdd : a PyramidNetAdd Module.
    """

    return PyramidNetAdd(pyramid_alpha = 48,
                         pyramid_depth = 164,
                         num_outputs = num_outputs, 
                         *args, **kwargs)


@_register_model("PyramidNet_164_270")
def PyramidNet_164_270(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-164-270 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetAdd : a PyramidNetAdd Module.
    """

    return PyramidNetAdd(pyramid_alpha = 270,
                         pyramid_depth = 164,
                         num_outputs = num_outputs, 
                         *args, **kwargs)


@_register_model("PyramidNet_200_240")
def PyramidNet_200_240(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-200-240 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetAdd : a PyramidNetAdd Module.
    """

    return PyramidNetAdd(pyramid_alpha = 240,
                         pyramid_depth = 200,
                         num_outputs = num_outputs, 
                         *args, **kwargs)


@_register_model("PyramidNet_272_200")
def PyramidNet_272_200(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-272-200 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetAdd : a PyramidNetAdd Module.
    """

    return PyramidNetAdd(pyramid_alpha = 200,
                         pyramid_depth = 272,
                         num_outputs = num_outputs, 
                         *args, **kwargs)

    
