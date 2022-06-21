from random import uniform
import jax
import flax.linen as nn
from typing import Tuple
from functools import partial
import jax.numpy as jnp
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

def shake_shake_train(xa, xb, rng = None, true_gradient = False):

    # if rng is None:
    #     rng = flax.nn.make_rng()
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

def shake_shake_eval(xa, xb):
    return (xa + xb) * 0.5


class ShortCut(nn.Module):

    channels : int
    strides : Tuple[int, int] = (1, 1)
    
    @nn.compact
    def __call__(self, x, train):

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

    channels : int
    strides : Tuple[int, int] = (1, 1)

    def setup(self):
        self.param("shake_params", jax.random.uniform, (1,))

    @nn.compact
    def __call__(self, x, train, true_gradient = False):
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

        # is_initialized = self.has_variable("batch_stats", "mean")
        # if train and not is_initialized:
        if train:
            rng = self.make_rng("shake")
            ab = shake_shake_train(a, b, true_gradient=true_gradient, rng = rng)
        else:
            ab = shake_shake_eval(a, b)

        residual = ShortCut(self.channels, self.strides)(residual, train = train)

        return residual + ab


class WideResNetShakeShakeGroup(nn.Module):

    blocks_per_group : int
    channels : int
    strides : Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, train, true_gradient = False):

        for i in range(self.blocks_per_group):
            x = ShakeShakeBlock(channels = self.channels,
                    strides = self.strides if i == 0 else (1, 1))(x, train = train, true_gradient=true_gradient)
        return x


class WideResNetShakeShake(nn.Module):

    blocks_per_group : int
    channel_multiplier: int
    num_outputs : int
    base_channel : int = 16

    @nn.compact
    def __call__(self, inputs, train, true_gradient = False):    
        
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


@_register_model("WideResNet_SS")
def WideResNet_SS(num_outputs, *args, **kwargs):

    return WideResNetShakeShake(blocks_per_group = 4,
                                channel_multiplier = 6,
                                num_outputs = num_outputs, 
                                *args, **kwargs)


def init_image_model(
	prng_key: jnp.ndarray, batch_size: int, image_size: int,
	module: flax.linen.Module,
	num_channels: int = 3):

	dummy_input = jnp.zeros(shape = (batch_size, image_size, image_size, num_channels))
	variables = module.init(
	    prng_key, dummy_input, train = False
	)
	state, params = variables.pop("params")
	
	return params, state


if __name__ == "__main__":
    model = WideResNetShakeShake(1, 4, 10)
    params, state = jax.jit(lambda : init_image_model(jax.random.PRNGKey(0), 10, 32, model, 3), backend = "cpu")()
    print(params)

