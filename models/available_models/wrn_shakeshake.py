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
    Wide ResNet Model with Shake-Shake Regularization.

    Reference:
      Shake-Shake regularization (ICLR (Workshop) 2018)
      [https://arxiv.org/abs/1705.07485]
    
    This implementation mimics that in
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/wide_resnet_shakeshake.py
    and
    https://github.com/google-research/sam.
"""

import jax
import flax.linen as nn
from typing import Tuple, Optional
from functools import partial
import jax.numpy as jnp
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model


def shake_shake_train(xa : jnp.ndarray,
                      xb : jnp.ndarray,
                      rng : jnp.ndarray,
                      true_gradient : Optional[bool] = False):
    """Shake-shake regularization in training mode.
    
    Forked from:
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/utils.py

    Shake-shake regularization interpolates between inputs A and B
    with *different* random uniform (per-sample) interpolation factors
    for the forward and backward/gradient passes.

    Args:
        xa: Input, branch A.
        xb: Input, branch B.
        rng: PRNG key.
        true_gradient: If true, the same mixing parameter will be used for the
                         forward and backward pass (see paper for more details).

    Returns:
        Mix of input branches.
    """

    gate_forward_key, gate_backward_key = jax.random.split(rng, num = 2)
    gate_shape = (len(xa), 1, 1, 1)

    gate_forward = jax.random.uniform(
        gate_forward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0
    )
    x_forward = xa * gate_forward + xb * (1.0 - gate_forward)
    if true_gradient:
        return x_forward

    gate_backward = jax.random.uniform(
        gate_backward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0
    )
    x_backward = xa * gate_backward + xb * (1.0 - gate_backward)
    return x_backward + jax.lax.stop_gradient(x_forward - x_backward)


def shake_shake_eval(xa : jnp.ndarray,
                     xb : jnp.ndarray,):
    """Shake-shake regularization in testing mode.

    Args:
        xa: Input, branch A.
        xb: Input, branch B.

    Returns:
        Mix of input branches.
    """

    return (xa + xb) * 0.5


class ShortCut(nn.Module):
    """
        ShortCut Class for residual connections.

        Attributes:
            channels :the number of channels to use in one convolutional
              layer.
            strides : the strides of convlution.
    """
    channels : int
    strides : Optional[Tuple[int, int]] = (1, 1)
    
    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply ShortCut module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block
        """

        if x.shape[-1] == self.channels:
            return x

        # Skip path 1
        h1 = nn.avg_pool(x, (1, 1), strides = self.strides, padding = "VALID")
        h1 = nn.Conv(self.channels // 2, (1, 1),
                    strides = (1, 1), padding = "SAME", use_bias = False,
                    kernel_init=conv_kernel_init)(h1)
        
        # Skip path 2
        pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
        h2 = jnp.pad(x, pad_arr)[:, 1:, 1:, :]
        h2 = nn.avg_pool(h2, (1, 1), strides=self.strides, padding='VALID')
        h2 = nn.Conv(self.channels // 2, (1, 1),
                    strides = (1, 1), padding = "SAME", use_bias = False,
                    kernel_init=conv_kernel_init)(h2)
        
        merged_branches = jnp.concatenate([h1, h2], axis=3)
        h = ActivationOp()(merged_branches, train = train, apply_relu=False)
        return h


class ShakeShakeBlock(nn.Module):
    """
        Shake-Shake Block Class.

        Attributes:
            channels :the number of channels to use in one convolutional
              layer.
            strides : the strides of convlution.
    """
    channels : int
    strides : Optional[Tuple[int, int]] = (1, 1)

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool,
                 true_gradient : bool = False):
        """
            Apply ShortCut module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                true_gradient : if true, the same mixing parameter will be used
                  forward and backward pass (see paper for more details).

            Returns:
                x : output of the block.
        """

        a = b = residual = x

        a = jax.nn.relu(a)
        a = nn.Conv(self.channels, (3, 3), strides = self.strides, padding = "SAME",
                    use_bias = False,  kernel_init=conv_kernel_init)(a)
        a = ActivationOp()(a, train = train)
        a = nn.Conv(self.channels, (3, 3), padding = "SAME",
                    use_bias = False,  kernel_init=conv_kernel_init)(a)
        a = ActivationOp()(a, train = train, apply_relu=False)

        b = jax.nn.relu(b)
        b = nn.Conv(self.channels, (3, 3), strides = self.strides, padding = "SAME",
                    use_bias = False,  kernel_init=conv_kernel_init)(b)
        b = ActivationOp()(b, train = train)
        b = nn.Conv(self.channels, (3, 3), padding = "SAME",
                    use_bias = False,  kernel_init=conv_kernel_init)(b)
        b = ActivationOp()(b, train = train, apply_relu=False)

        if train:
            # Make rng and pass to shake_shake function.
            rng = self.make_rng("shake")
            ab = shake_shake_train(a, b, true_gradient=true_gradient, rng = rng)
        else:
            ab = shake_shake_eval(a, b)

        residual = ShortCut(self.channels, self.strides)(residual, train = train)

        return residual + ab


class WideResNetShakeShakeGroup(nn.Module):
    """
        Wide ResNet Shake-Shake Group Class.

        Attributes:
            blocks_per_group :the number of shake-shake block to add to each
              block.
            channels : the number of channels to use in one convolutional
              layer.
            strides : the strides of convlution.
    """

    blocks_per_group : int
    channels : int
    strides : Optional[Tuple[int, int]] = (1, 1)

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool,
                 true_gradient : bool = False):
        """
            Apply Wide ResNet Shake-Shake Group module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                true_gradient : if true, the same mixing parameter will be used
                  forward and backward pass (see paper for more details).

            Returns:
                x : output of the block.
        """
        
        for i in range(self.blocks_per_group):
            x = ShakeShakeBlock(channels = self.channels,
                    strides = self.strides if i == 0 else (1, 1))(x, train = train, true_gradient=true_gradient)
        return x


class WideResNetShakeShake(nn.Module):
    """
        Wide ResNet Shake-Shake Group Class.

        Attributes:
            blocks_per_group :the number of shake-shake block to add to each
              block.
            channel_multiplier : the multiplier to apply to the number of
              filters in the model.
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
            base_channel : the base number of channels of the model. The total
              number for a block would be base_channel * channel_multiplier.
    """
    
    blocks_per_group : int
    channel_multiplier: int
    num_outputs : int
    base_channel : Optional[int] = 16

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool,
                 true_gradient : bool = False):
        """
            Apply Wide ResNet Shake-Shake module.

            Args:
                inputs : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                true_gradient : if true, the same mixing parameter will be used
                  forward and backward pass (see paper for more details).

            Returns:
                x : output of the block.
        """
        
        x = inputs
        x = nn.Conv(16, (3, 3), padding = "SAME", use_bias = False,
                 kernel_init=conv_kernel_init)(x)
        
        x = ActivationOp()(x, apply_relu=False, train = train)
        x = WideResNetShakeShakeGroup(blocks_per_group = self.blocks_per_group,
                        channels = self.base_channel * self.channel_multiplier)(x, train = train, true_gradient=true_gradient)

        x = WideResNetShakeShakeGroup(blocks_per_group = self.blocks_per_group,
                        channels = self.base_channel * 2 * self.channel_multiplier,
                        strides = (2, 2))(x, train = train, true_gradient=true_gradient)

        x = WideResNetShakeShakeGroup(blocks_per_group = self.blocks_per_group,
                        channels = self.base_channel * 4 * self.channel_multiplier,
                        strides = (2, 2))(x, train = train, true_gradient=true_gradient)

        x = jax.nn.relu(x)
        x = nn.avg_pool(x, x.shape[1:3])
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
        return x


@_register_model("WideResNet_2_96_ShakeShake")
def WideResNet_2_96_ShakeShake(num_outputs : int, *args, **kwargs):
    """
        Build WideResNet-2-96 module. 2 denotes the number of shake-shake
          branches and 96 represents the width of the first residual block.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            WideResNetShakeShake : a WideResNetShakeShake Module.
    """
    
    return WideResNetShakeShake(blocks_per_group = 4,
                                channel_multiplier = 6,
                                num_outputs = num_outputs, 
                                *args, **kwargs)


