from typing import Tuple

from absl import flags
import flax.linen as nn
import jax
from jax import numpy as jnp
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


class VGG16_BN(nn.Module):

    num_outputs: int 

    @nn.compact
    def __call__(self,
              inputs: jnp.ndarray,
              train: bool = True) -> jnp.ndarray:

        x = inputs
        x = nn.Conv(64, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block1_conv1')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(64, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block1_conv2')(x)
        x = ActivationOp()(x, train = train)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(128, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block2_conv1')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(128, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block2_conv2')(x)
        x = ActivationOp()(x, train = train)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block3_conv1')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block3_conv2')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block3_conv3')(x)
        x = ActivationOp()(x, train = train)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block4_conv1')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block4_conv2')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block4_conv3')(x)
        x = ActivationOp()(x, train = train)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block5_conv1')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block5_conv2')(x)
        x = ActivationOp()(x, train = train)
        x = nn.Conv(512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True, kernel_init=conv_kernel_init, name='block5_conv3')(x)
        x = ActivationOp()(x, train = train)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(4096, kernel_init=dense_layer_init_fn, name='fc1')(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)

        x = nn.Dense(4096, kernel_init=dense_layer_init_fn, name='fc2')(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)

        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)

        return x


class VGG16(nn.Module):

    def apply(self,
              x: jnp.ndarray,
              num_outputs: int,
              train: bool = True) -> jnp.ndarray:

        x = nn.Conv(x, 64, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block1_conv1')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 64, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block1_conv2')
        x = jax.nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(x, 128, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block2_conv1')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 128, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block2_conv2')
        x = jax.nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(x, 256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block3_conv1')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block3_conv2')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 256, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block3_conv3')
        x = jax.nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block4_conv1')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block4_conv2')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block4_conv3')
        x = jax.nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block5_conv1')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block5_conv2')
        x = jax.nn.relu(x)
        x = nn.Conv(x, 512, (3, 3), strides = (1, 1), padding = "SAME",
                   use_bias = True,   name='block5_conv3')
        x = jax.nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(x, 4096,  name='fc1')
        x = jax.nn.relu(x)
        x = nn.dropout(x, 0.5, deterministic=not train)

        x = nn.Dense(x, 4096,  name='fc2')
        x = jax.nn.relu(x)
        x = nn.dropout(x, 0.5, deterministic=not train)

        x = nn.Dense(x, num_outputs,)

        return x


@_register_model("VGG16BN")
def VGG16BN(num_outputs, *args, **kwargs):
  return VGG16_BN(num_outputs = num_outputs,
                 *args, **kwargs)
