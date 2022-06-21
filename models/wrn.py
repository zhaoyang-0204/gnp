import jax
import flax.linen as nn
from typing import Tuple
from functools import partial
import jax.numpy as jnp

from gnp.models import _register_model

conv_kernel_init = jax.nn.initializers.variance_scaling(2.0, "fan_out", "normal")
_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5

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


def _output_add(block_x, orig_x):

    stride = orig_x.shape[-2] // block_x.shape[-2]
    strides = (stride, stride)
    if block_x.shape[-1] != orig_x.shape[-1]:
        orig_x = nn.avg_pool(orig_x, strides, strides)
        channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
        orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
    return block_x + orig_x

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


class WideResNetBlock(nn.Module):

    channels: int
    strides: Tuple[int, int] = (1, 1)
    activation_before_residual: bool = False

    @nn.compact
    def __call__(self, x, train):

        if self.activation_before_residual:
            x = ActivationOp()(x, train = train)
            orig_x = x
        else:
            orig_x = x

        block_x = x
        if not self.activation_before_residual:
            block_x = ActivationOp()(block_x, train = train)

        block_x = nn.Conv(
            self.channels, (3, 3), self.strides, padding = "SAME", use_bias = False,
            kernel_init=conv_kernel_init, name = "conv1"
        )(block_x)

        block_x = ActivationOp()(block_x, train = train)

        block_x = nn.Conv(
            self.channels, (3, 3), padding = "SAME", use_bias = False,
            kernel_init=conv_kernel_init, name = "conv2"
        )(block_x)

        return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):

    block_per_group: int
    channels: int
    strides: Tuple[int, int] = (1, 1)
    activation_before_residual: bool = False

    @nn.compact
    def __call__(self, x, train):
        for i in range(self.block_per_group):
            x = WideResNetBlock(
                self.channels, self.strides if i == 0 else (1, 1),
                activation_before_residual=self.activation_before_residual and not i,
                )(x, train = train)
        return x


class WideResNet(nn.Module):

    block_per_group: int
    channel_multiplier: int
    num_outputs: int 

    @nn.compact
    def __call__(self, inputs, train):
        
        x = nn.Conv(16,
                    (3, 3),
                    padding="SAME",
                    name = "init_conv",
                    kernel_init=conv_kernel_init,
                    use_bias = False)(inputs)

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

        x = ActivationOp()(x, train = train)
        x = nn.avg_pool(x, x.shape[1:3])
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
        return x
    

@_register_model("WideResNet_28_10")
def WideResNet_28_10(num_outputs, *args, **kwargs):

    return WideResNet(block_per_group = 4,
                      channel_multiplier = 10,
                      num_outputs = num_outputs, 
                      *args, **kwargs)


        
