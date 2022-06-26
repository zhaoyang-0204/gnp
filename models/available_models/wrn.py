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
    Wide ResNet Model.

    Reference:
      Wide Residual Networks (BMVC 2016)
      [https://arxiv.org/abs/1605.07146]

    This implementation mimics that in
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/wide_resnet.py
    and
    https://github.com/google-research/sam.

    From the SAM repo [https://github.com/google-research/sam]:
    "
        It uses idendity + zero padding skip connections, with kaiming normal
        initialization for convolutional kernels (mode = fan_out, gain=2.0).
        The final dense layer use a uniform distribution U[-scale, scale] where
        scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

        Using the default initialization instead gives error rates approximately 0.5%
        greater on cifar100, most likely between the parameters used in the literature
        where finetuned for this particular initialization.

        Finally, the autoaugment implementation adds more residual connections between
        the groups (instead of just between the blocks as per the original paper and
        most implementations). It is possible to safely remove those connections without
        degrading the performances, which we do by default to match the original
        wideresnet paper. Setting `use_additional_skip_connections` to True will add
        them back and then reproduces exactly the model used in autoaugment.
    "
"""

import flax.linen as nn
from typing import Tuple, Optional
import jax.numpy as jnp
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model
from absl import flags
FLAGS = flags.FLAGS


def _output_add(block_x : jnp.ndarray,
               orig_x : jnp.ndarray) -> jnp.ndarray:
    """Add two tensors, padding them with zeros or pooling them if necessary.

    Args:
        block_x: Output of a resnet block.
        orig_x: Residual branch to add to the output of the resnet block.

    Returns:
        The sum of blocks_x and orig_x. If necessary, orig_x will be average
        pooled or zero padded so that its shape matches orig_x.
    """

    stride = orig_x.shape[-2] // block_x.shape[-2]
    strides = (stride, stride)
    if block_x.shape[-1] != orig_x.shape[-1]:
        orig_x = nn.avg_pool(orig_x, strides, strides)
        channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
        orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
    return block_x + orig_x


class WideResNetBlock(nn.Module):

    """
        Wide ResNet Block Class.

        Attributes:
            filters : the number of filters to use in one convolutional
              layer.
            strides : the strides of convlution.
            activation_before_residual : True if the batch norm and relu should
              be applied before the residual branches out (should be True only for
              the first block of the model).
    """

    filters: int
    strides: Optional[Tuple[int, int]] = (1, 1)
    activation_before_residual: Optional[bool] = False

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply Wide ResNet Block module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block.
        """

        if self.activation_before_residual:
            x = ActivationOp()(x, train = train)
            orig_x = x
        else:
            orig_x = x

        block_x = x
        if not self.activation_before_residual:
            block_x = ActivationOp()(block_x, train = train)

        block_x = nn.Conv(
            self.filters, (3, 3), self.strides, padding = "SAME", use_bias = False,
            kernel_init=conv_kernel_init, name = "conv1"
        )(block_x)

        block_x = ActivationOp()(block_x, train = train)

        block_x = nn.Conv(
            self.filters, (3, 3), padding = "SAME", use_bias = False,
            kernel_init=conv_kernel_init, name = "conv2"
        )(block_x)

        return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):

    """
        Wide ResNet Group Class.

        Attributes:
            block_per_group : the number of resnet blocks to add to each group
              (should be 4 blocks for a WRN28, and 6 for a WRN40).
            filters : the number of filters to use in one convolutional
              layer.
            strides : the strides of convlution.
            activation_before_residual : True if the batch norm and relu should
              be applied before the residual branches out (should be True only
              for the first block of the model).
    """

    block_per_group: int
    filters: int
    strides: Optional[Tuple[int, int]] = (1, 1)
    activation_before_residual: Optional[bool] = False

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply Wide ResNet Group module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
        """
        orig_x = x
        for i in range(self.block_per_group):
            x = WideResNetBlock(
                self.filters, self.strides if i == 0 else (1, 1),
                activation_before_residual=self.activation_before_residual and not i,
                )(x, train = train)
        if FLAGS.config.use_additional_skip_connections_in_wrn:
            x = _output_add(x, orig_x)

        return x


class WideResNet(nn.Module):

    """
        Wide ResNet Group Class.

        Attributes:
            block_per_group : the number of resnet blocks to add to each group
              (should be 4 blocks for a WRN28, and 6 for a WRN40).
            channel_multiplier : the multiplier to apply to the number of
              filters in the model (1 is classical resnet, 10 for WRN28-10,
              etc...).
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
    """

    block_per_group: int
    channel_multiplier: int
    num_outputs: int 

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool):
        """
            Wide ResNet module.

            Args:
                inputs : inputs to the module.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : outputs of the module.
        """

        orig_x = x = inputs
        x = nn.Conv(16,
                    (3, 3),
                    padding="SAME",
                    name = "init_conv",
                    kernel_init=conv_kernel_init,
                    use_bias = False)(x)

        x = WideResnetGroup(
            self.block_per_group, self.channel_multiplier * 16,
            activation_before_residual=True,
        )(x, train = train)

        x = WideResnetGroup(
            self.block_per_group, self.channel_multiplier * 32, 
            strides = (2, 2)
        )(x, train = train)

        x = WideResnetGroup(
            self.block_per_group, self.channel_multiplier * 64,
            strides = (2, 2)
        )(x, train = train)

        if FLAGS.config.use_additional_skip_connections_in_wrn:
            x = _output_add(x, orig_x)

        x = ActivationOp()(x, train = train)
        x = nn.avg_pool(x, x.shape[1:3])
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
        return x


@_register_model("WideResNet_16_4")
def WideResNet_16_4(num_outputs : int, *args, **kwargs):
    """
        Build WideResNet-16-4 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            WideResNet : a WideResNet Module.
    """

    return WideResNet(block_per_group = 2,
                      channel_multiplier = 4,
                      num_outputs = num_outputs, 
                      *args, **kwargs)


@_register_model("WideResNet_28_10")
def WideResNet_28_10(num_outputs : int, *args, **kwargs):
    """
        Build WideResNet-28-10 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            WideResNet : a WideResNet Module.
    """

    return WideResNet(block_per_group = 4,
                      channel_multiplier = 10,
                      num_outputs = num_outputs, 
                      *args, **kwargs)


@_register_model("WideResNet_40_10")
def WideResNet_40_10(num_outputs : int, *args, **kwargs):
    """
        Build WideResNet-40-10 module.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            WideResNet : a WideResNet Module.
    """

    return WideResNet(block_per_group = 6,
                      channel_multiplier = 10,
                      num_outputs = num_outputs, 
                      *args, **kwargs)

        
