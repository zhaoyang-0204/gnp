import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple
from functools import partial

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


class BottleneckBlock(nn.Module):

    channels : int
    strides : int

    @nn.compact
    def __call__(self, x, train):
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

    pyramid_alpha : int
    pyramid_depth : int
    num_outputs : int

    @nn.compact
    def __call__(self, x, train):

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
def PyramidNet_164_48(num_outputs, *args, **kwargs):
    return PyramidNetAdd(pyramid_alpha = 48,
                         pyramid_depth = 164,
                         num_outputs = num_outputs, 
                         *args, **kwargs)


@_register_model("PyramidNet_164_270")
def PyramidNet_164_270(num_outputs, *args, **kwargs):

    return PyramidNetAdd(pyramid_alpha = 270,
                         pyramid_depth = 164,
                         num_outputs = num_outputs, 
                         *args, **kwargs)


@_register_model("PyramidNet_200_240")
def PyramidNet_200_240(num_outputs, *args, **kwargs):

    return PyramidNetAdd(pyramid_alpha = 240,
                         pyramid_depth = 200,
                         num_outputs = num_outputs, 
                         *args, **kwargs)

@_register_model("PyramidNet_270_200")
def PyramidNet_270_200(num_outputs, *args, **kwargs):

    return PyramidNetAdd(pyramid_alpha = 200,
                         pyramid_depth = 270,
                         num_outputs = num_outputs, 
                         *args, **kwargs)



if __name__ == "__main__":
    model = PyramidNet_164_48(num_outputs = 10)
    
