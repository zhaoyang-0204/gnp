import jax
import flax.linen as nn
from typing import Tuple, List
from functools import partial
import jax.numpy as jnp

conv_kernel_init = jax.nn.initializers.variance_scaling(2.0, "fan_out", "normal")
_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5

from gnp.models import _register_model


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


class ResNetBlock(nn.Module):

    filters : int
    kernel_size : Tuple[int, int] = (3, 3)
    strides : Tuple[int, int] = (1, 1)
    conv_shortcut : bool = True

    @nn.compact
    def __call__(self, x, train):
        
        if self.conv_shortcut:
            shortcut = nn.Conv(4 * self.filters, kernel_size = (1, 1),
                              strides = self.strides)(x)
            shortcut = ActivationOp()(shortcut, apply_relu = False, train = train)
        else:
            shortcut = x

        x = nn.Conv(self.filters, kernel_size=(1, 1),
                         strides = self.strides)(x)
        x = ActivationOp()(x, train = train)

        x = nn.Conv(self.filters, kernel_size=self.kernel_size,
                         )(x)
        x = ActivationOp()(x, train = train)

        x = nn.Conv(4 * self.filters, kernel_size=(1, 1),
                         )(x)
        x = ActivationOp()(x, apply_relu = False, train = train)

        x = shortcut + x
        x = jax.nn.relu(x)

        return x

class ResNetStack(nn.Module):

    filters : int
    blocks : int
    strides : Tuple[int, int] = (2, 2)

    @nn.compact
    def __call__(self, x, train):
        x = ResNetBlock(filters = self.filters,
                        strides=self.strides)(x, train = train)

        for i in range(2, self.blocks + 1):
            x = ResNetBlock(filters = self.filters,
                            conv_shortcut=False)(x, train = train)

        return x

class BottleneckResNetBlock(nn.Module):

    filters : int
    strides : Tuple[int, int]

    @nn.compact
    def __call__(self, x, train):
        
        # if self.conv_shortcut:
        #     shortcut = nn.Conv(4 * self.filters, kernel_size = (1, 1),
        #                       strides = self.strides)(x)
        #     shortcut = ActivationOp()(shortcut, apply_relu = False, train = train)
        # else:
        #     shortcut = x
        residual = x
        y = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False
                         )(x)
        y = ActivationOp()(y, train = train)

        y = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False, 
                        strides =  self.strides)(y)
        y = ActivationOp()(y, train = train)

        y = nn.Conv(4 * self.filters, kernel_size=(1, 1), use_bias=False, 
                         )(y)
        y = ActivationOp()(y, apply_relu = False, train = train)

        if residual.shape != y.shape:
            residual = nn.Conv(4 * self.filters, kernel_size=(1, 1), use_bias=False, strides = self.strides
                         )(residual)
            residual = ActivationOp()(residual, apply_relu = False, train = train)

        y = residual + y
        y = jax.nn.relu(y)
        return y


class ResNet(nn.Module):
    
    block_sizes: List
    num_outputs: int
    num_filters : int = 64
    preact : bool = False
    use_bias : bool = True
    dtype : jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train):
        x = inputs
        # x = jnp.pad(x,  [(0, 0), (3, 3), (3, 3), (0, 0)])
        if self.num_outputs == 1000:
            print("ResNet with Imagenet")
            x = nn.Conv(self.num_filters, (7, 7), strides = (2, 2),
                        padding="SAME", use_bias=False
                        )(x)   
            x = ActivationOp()(x, apply_relu = False, train = train)
            # x = jnp.pad(x,  [(0, 0), (1, 1), (1, 1), (0, 0)])
            x = nn.max_pool(x, (3, 3), strides = (2, 2), padding='SAME')
        elif self.num_outputs in (10, 100):
            print("ResNet with Cifar")
            x = nn.Conv(16, (3, 3), strides = (1, 1),
                        padding="SAME", use_bias=False
                        )(x)
            x = ActivationOp()(x, apply_relu = False, train = train)            

        # Model Core
        for i, block_size in enumerate(self.block_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = BottleneckResNetBlock(filters = self.num_filters * 2 ** i, 
                                          strides = strides,
                                        )(x, train = train)

        x = jnp.mean(x, axis = (1, 2))
        x = nn.Dense(self.num_outputs, dtype = self.dtype)(x)
   
        return x


@_register_model("ResNet50")
def ResNet50(num_outputs, *args, **kwargs):

    return ResNet(block_sizes = [3, 4, 6, 3],
                num_outputs = num_outputs, 
                *args, **kwargs)

@_register_model("ResNet101")
def ResNet101(num_outputs, *args, **kwargs):

    return ResNet(block_sizes = [3, 4, 23, 3],
                num_outputs = num_outputs, 
                *args, **kwargs)


if __name__ == "__main__":
    model = ResNet50(10)
    print(model)