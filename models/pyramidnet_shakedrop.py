import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple
from functools import partial
import flax

from gnp.models import _register_model

conv_kernel_init = jax.nn.initializers.variance_scaling(2.0, "fan_out", "normal")
_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5

def dense_layer_init_fn(key: jnp.ndarray,
                        shape: Tuple[int, int],
                        dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
  """Initializer for the final dense layer.

  Args:
    key: PRNG key to use to sample the weights.
    shape: Shape of the tensor to initialize.
    dtype: Data type of the tensor to initialize.

  Returns:
    The initialized tensor.
  """
  num_units_out = shape[1]
  unif_init_range = 1.0 / (num_units_out)**(0.5)
  return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


class ActivationOp(nn.Module):

    @nn.compact
    def __call__(self, x, train, apply_relu = True):
        norm = partial(nn.BatchNorm, use_running_average = not train,
                            momentum = _BATCHNORM_MOMENTUM, epsilon = _BATCHNORM_EPSILON,
                            ) 
        x = norm()(x)
        if apply_relu:
            x = jax.nn.relu(x)

        return x


def _shortcut(x, chn_out, strides):

    chn_in = x.shape[3]
    if strides != (1, 1):
        x = nn.avg_pool(x, strides, strides)
    if chn_out != chn_in:
        diff = chn_out - chn_in
        x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [0, diff]])
    return x


def shake_drop_train(x, mask_prob, alpha_min, alpha_max, beta_min, beta_max,
                    rng = None, true_gradient = False):

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

def shake_drop_eval(x, mask_prob, alpha_min, alpha_max):

    expected_alpha = (alpha_min + alpha_max) / 2
    return (mask_prob + expected_alpha - mask_prob * expected_alpha) * x


def _calc_shakedrop_mask_prob(curr_layer : int,
                              total_layers : int,
                              mask_prob : float):

    return 1 - (float(curr_layer) / total_layers) * mask_prob


class BottleneckShakeDropBlock(nn.Module):

    channels : int
    strides : Tuple[int, int]
    prob : float
    alpha_min : float
    alpha_max : float
    beta_min : float
    beta_max : float
    
    @nn.compact
    def __call__(self, x, train, true_gradient = False):   
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
            x = shake_drop_train(x, self.prob, self.alpha_min, self.alpha_max,
                                self.beta_min, self.beta_max,
                                true_gradient = true_gradient, rng = rng)
        else:
            x = shake_drop_eval(x, self.prob, self.alpha_min, self.alpha_max)

        residual = _shortcut(residual, self.channels * 4, self.strides)
        return x + residual


class PyramidNetShakeDrop(nn.Module):

    pyramid_alpha : int
    pyramid_depth : int
    num_outputs : int
    mask_prob : float = 0.5
    alpha : Tuple[float, float] = (-1.0, 1.0)
    beta : Tuple[float, float] = (0.0, 1.0)

    @nn.compact
    def __call__(self, inputs, train, true_gradient = False):
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

@_register_model("PyramidNet_272_200_SD")
def PyramidNet_272_200_SS(num_outputs, *args, **kwargs):

    return PyramidNetShakeDrop(pyramid_alpha = 200,
                               pyramid_depth = 272,
                               num_outputs = num_outputs, 
                               *args, **kwargs)