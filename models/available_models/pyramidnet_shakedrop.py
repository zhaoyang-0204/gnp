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
    PyramidNet Model with Shake-Drop Regularization. 

    Reference:
      Deep Pyramidal Residual Networks.
      [https://arxiv.org/pdf/1610.02915.pdf]
      ShakeDrop Regularization for Deep Residual Learning.
      [https://arxiv.org/abs/1802.02375]

    This implementation mimics that in
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/wide_resnet_shakeshake.py
    and
    https://github.com/google-research/sam.
"""

import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Optional
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model
from gnp.models.available_models.pyramidnet import _shortcut


def shake_drop_train(x : jnp.ndarray,
                     mask_prob : float,
                     alpha_min : float,
                     alpha_max : float,
                     beta_min : float,
                     beta_max : float,
                     rng : jnp.ndarray,
                     true_gradient : Optional[bool] = False):
    """
    Shake-drop regularization in training mode. See paper for details.
    
    Forked from:
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/utils.py

    Args:
        x : input to the shake-drop regularization.
        mask_prob : mask probability.
        alpha_min : the minimum in the range of parameter alpha.
        alpha_max : the maximum in the range of parameter alpha.
        beta_min : the minimum in the range of parameter beta.
        beta_max : the maximum in the range of parameter beta.
        rng : PRNG key.
        true_gradient: If true, the same mixing parameter will be used for the
                         forward and backward pass.
    
    Returns:
        output.
    """

    bern_key, alpha_key, beta_key = jax.random.split(rng, num = 3)
    rnd_shape = (len(x), 1, 1, 1)
    mask = jax.random.bernoulli(bern_key, mask_prob, rnd_shape)
    mask = mask.astype(jnp.float32)
    alpha_values = jax.random.uniform(
        alpha_key,
        rnd_shape,
        dtype=jnp.float32,
        minval=alpha_min,
        maxval=alpha_max)
    beta_values = jax.random.uniform(
        beta_key, rnd_shape, dtype=jnp.float32, minval=beta_min, maxval=beta_max)
    rand_forward = mask + alpha_values - mask * alpha_values
    if true_gradient:
        return x * rand_forward
    rand_backward = mask + beta_values - mask * beta_values
    return x * rand_backward + jax.lax.stop_gradient(
        x * rand_forward - x * rand_backward)


def shake_drop_eval(x : jnp.ndarray,
                    mask_prob : float,
                    alpha_min : float,
                    alpha_max : float):
    """
    Shake-drop regularization in training mode. See paper for details.
    
    Forked from:
    https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/utils.py

    Args:
        x : input to the shake-drop regularization.
        mask_prob : mask probability.
        alpha_min : the minimum in the range of parameter alpha.
        alpha_max : the maximum in the range of parameter alpha.
    
    Returns:
        output.
    """
    expected_alpha = (alpha_min + alpha_max) / 2
    return (mask_prob + expected_alpha - mask_prob * expected_alpha) * x


def _calc_shakedrop_mask_prob(curr_layer : int,
                              total_layers : int,
                              mask_prob : float):
    """Calculates drop prob depending on the current layer."""

    return 1 - (float(curr_layer) / total_layers) * mask_prob


class BottleneckShakeDropBlock(nn.Module):
    """
        Bottleneck Shake-Drop Class.

        Attributes:
            channels :the number of channels to use in one convolutional
              layer.
            strides : the strides of convlution.
            mask_prob : the probability of dropping the block.
            alpha_min : the minimum in the range of parameter alpha.
            alpha_max : the maximum in the range of parameter alpha.
            beta_min : the minimum in the range of parameter beta.
            beta_max : the maximum in the range of parameter beta.            
    """

    channels : int
    strides : Tuple[int, int]
    mask_prob : float
    alpha_min : float
    alpha_max : float
    beta_min : float
    beta_max : float
    
    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool,
                 true_gradient : bool = False):
        """
            Apply Bottleneck Shake-Drop module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                true_gradient : if true, the same mixing parameter will be used
                  forward and backward pass (see paper for more details).

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

        if train:
            rng = self.make_rng("shake")
            x = shake_drop_train(x, self.mask_prob, self.alpha_min, self.alpha_max,
                                self.beta_min, self.beta_max,
                                true_gradient = true_gradient, rng = rng)
        else:
            x = shake_drop_eval(x, self.mask_prob, self.alpha_min, self.alpha_max)

        residual = _shortcut(residual, self.channels * 4, self.strides)
        return x + residual


class PyramidNetShakeDrop(nn.Module):
    """
        PyramidNet Shake-Drop module.

        Attributes:
            pyramid_alpha : the increase factor of channels, whihc is 
              D_k+1 = D_k + alpha/total_blocks.
            pyramid_depth : the total depth of the convolutional layers. For
              bottleneck structure, the depth - 2 must be divisible by 9.
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
            mask_prob : the probability of dropping the block.
            alpha : the range of shake-drop parameter alpha.
            beta : the range of shake-dropparameter beta.
    """

    pyramid_alpha : int
    pyramid_depth : int
    num_outputs : int
    mask_prob : float = 0.5
    alpha : Tuple[float, float] = (-1.0, 1.0)
    beta : Tuple[float, float] = (0.0, 1.0)


    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool,
                 true_gradient : bool = False):
        """
            Apply PyramindNet Shake-Drop module.

            Args:
                inputs: inputs to the module.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                true_gradient : if true, the same mixing parameter will be used
                  forward and backward pass (see paper for more details).

            Returns:
                x : output of the block.
        """  

        alpha_min, alpha_max = self.alpha
        beta_min, beta_max = self.beta
        assert (self.pyramid_depth - 2) % 9 == 0 # Bottlenet Arch
        blocks_per_group = (self.pyramid_depth - 2) // 9 # //6 if basic block
        num_channels = 16
        total_blocks = blocks_per_group * 3
        delta_channels = self.pyramid_alpha / total_blocks    

        x = inputs
        x = nn.Conv(16, (3, 3), padding = "SAME",
                use_bias = False, kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train, apply_relu=False)

        layer_num = 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks,
                                                  self.mask_prob)
            x = BottleneckShakeDropBlock(channels = int(round(num_channels)), strides = (1, 1),
                        prob = layer_mask_prob, alpha_min = alpha_min, alpha_max = alpha_max,
                        beta_min = beta_min, beta_max = beta_max)(x, train = train, true_gradient = true_gradient)
            layer_num += 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks,
                                                  self.mask_prob)
            x = BottleneckShakeDropBlock(channels = int(round(num_channels)),
                         strides = (2, 2) if block_i == 0 else (1, 1),
                        prob = layer_mask_prob, alpha_min = alpha_min, alpha_max = alpha_max,
                        beta_min = beta_min, beta_max = beta_max)(x, train = train, true_gradient = true_gradient)
            layer_num += 1

        for block_i in range(blocks_per_group):
            num_channels += delta_channels
            layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks,
                                                  self.mask_prob)
            x = BottleneckShakeDropBlock(channels = int(round(num_channels)),
                         strides = (2, 2) if block_i == 0 else (1, 1),
                        prob = layer_mask_prob, alpha_min = alpha_min, alpha_max = alpha_max,
                        beta_min = beta_min, beta_max = beta_max)(x, train = train, true_gradient = true_gradient)
            layer_num += 1

        assert layer_num - 1 == total_blocks
        x = ActivationOp()(x, train = train)
        x = nn.avg_pool(x, (8, 8))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
        return x


@_register_model("PyramidNet_164_48_ShakeDrop")
def PyramidNet_164_48_ShakeDrop(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-164-48 with Shake-Drop module. 

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetShakeDrop : a PyramidNetShakeDrop Module.
    """

    return PyramidNetShakeDrop(pyramid_alpha = 48,
                               pyramid_depth = 164,
                               num_outputs = num_outputs, 
                               *args, **kwargs)


@_register_model("PyramidNet_272_200_ShakeDrop")
def PyramidNet_272_200_ShakeDrop(num_outputs : int, *args, **kwargs):
    """
        Build PyramidNet-272-200 with Shake-Drop module. 

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            PyramidNetShakeDrop : a PyramidNetShakeDrop Module.
    """

    return PyramidNetShakeDrop(pyramid_alpha = 200,
                               pyramid_depth = 272,
                               num_outputs = num_outputs, 
                               *args, **kwargs)