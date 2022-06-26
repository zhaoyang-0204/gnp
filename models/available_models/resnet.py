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
    ResNet Model.

    Reference:
      Deep Residual Learning for Image Recognition (CVPR 2015)
      [https://arxiv.org/abs/1512.03385]

    This implementation mimics that for training ImageNet in
    https://github.com/google/flax/blob/master/examples/imagenet/resnet_v1.py
    and
    https://github.com/google-research/sam.

    For training Cifar, the first convolutional layer would be different 
     where we would follow the convention 16 filters with (3, 3) kernel size.
"""

import jax
import flax.linen as nn
from typing import Tuple, List, Optional
from functools import partial
from absl import logging
import jax.numpy as jnp
from gnp.models.available_models.util import \
   conv_kernel_init, dense_layer_init_fn, ActivationOp
from gnp.models import _register_model


class ResNetBlock(nn.Module):

    """
        ResNet Block Class.

        Attributes:
            filters :the number of filters to use in one convolutional
              layer.
            strides : the strides of convlution.
    """

    filters : int
    strides : Optional[Tuple[int, int]] = (1, 1)

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply ResNet Block module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block
        """
        
        residual = x
        x = nn.Conv(self.filters, kernel_size=(3, 3),
                         strides = self.strides, kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train)

        x = nn.Conv(self.filters, kernel_size=(3, 3),
                         kernel_init=conv_kernel_init)(x)
        x = ActivationOp()(x, train = train, apply_relu = False,)

        if residual.shape != x.shape:
            residual = nn.Conv(self.filters, kernel_size = (1, 1),
                              strides = self.strides, kernel_init=conv_kernel_init)(residual)
            residual = ActivationOp()(residual, apply_relu = False, train = train)

        x = residual + x
        x = jax.nn.relu(x)

        return x


class BottleneckResNetBlock(nn.Module):

    """
        Bottleneck ResNet Block Class.

        Attributes:
            filters :the number of filters to use in one convolutional
              layer.
            strides : the strides of convlution.
    """

    filters : int
    strides : Tuple[int, int]

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool):
        """
            Apply Bottleneck ResNet Block module.

            Args:
                x : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block
        """

        residual = x
        y = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False,
                         kernel_init=conv_kernel_init)(x)
        y = ActivationOp()(y, train = train)

        y = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False, 
                        strides =  self.strides, kernel_init=conv_kernel_init)(y)
        y = ActivationOp()(y, train = train)

        y = nn.Conv(4 * self.filters, kernel_size=(1, 1), use_bias=False, 
                         kernel_init=conv_kernel_init)(y)
        y = ActivationOp()(y, apply_relu = False, train = train)

        if residual.shape != y.shape:
            residual = nn.Conv(4 * self.filters, kernel_size=(1, 1), use_bias=False,
                         strides = self.strides, kernel_init=conv_kernel_init)(residual)
            residual = ActivationOp()(residual, apply_relu = False, train = train)

        y = residual + y
        y = jax.nn.relu(y)
        return y


class ResNet(nn.Module):

    """
        Bottleneck ResNet Block Class.

        Attributes:
            block_sizes : the list collects how many convolutional layers would
              be applied in each block.
            block_cls : the block class to use whether a norm resnet block or
              the bottlenet block.
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
            num_filters : the base number of filters to use in the blocks. The
              number of filters would increase by a factor of 2 ** block_index.
            use_bias : if True, bias would be added additionally after performing
              convolution.
    """

    block_sizes: List
    block_cls : nn.Module
    num_outputs: int
    num_filters : Optional[int] = 64
    use_bias : Optional[bool] = True

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool):
        """
            Apply Bottleneck ResNet Block module.

            Args:
                inputs : inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.

            Returns:
                x : output of the block
        """
        x = inputs
        
        # The first convolutional layer would be differnt between training
        # ImageNet and Cifar
        if self.num_outputs == 1000:
            logging.info("ImageNet ResNet Module Arch")
            x = nn.Conv(self.num_filters, (7, 7), strides = (2, 2),
                        padding="SAME", use_bias=False, 
                        kernel_init=conv_kernel_init)(x)   
            x = ActivationOp()(x, apply_relu = False, train = train)
            x = nn.max_pool(x, (3, 3), strides = (2, 2), padding='SAME')
        elif self.num_outputs in (10, 100):
            logging.info("Cifar ResNet Module Arch")
            x = nn.Conv(16, (3, 3), strides = (1, 1),
                        padding="SAME", use_bias=False,
                        kernel_init=conv_kernel_init)(x)
            x = ActivationOp()(x, apply_relu = False, train = train)            

        # Model Core
        for i, block_size in enumerate(self.block_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(filters = self.num_filters * 2 ** i, 
                                          strides = strides,
                                        )(x, train = train)

        x = jnp.mean(x, axis = (1, 2))
        x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)

        return x


@_register_model("ResNet18")
def ResNet18(num_outputs : int, *args, **kwargs):
    """
        Build ResNet18 module.
        block_list = [2, 2, 2, 2], block_cls = ResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by ResNetBlock.
    """

    return ResNet(block_sizes = [2, 2, 2, 2],
                block_cls=ResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)

@_register_model("ResNet34")
def ResNet34(num_outputs : int, *args, **kwargs):
    """
        Build ResNet34 module.
        block_list = [3, 4, 6, 3], block_cls = ResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by ResNetBlock.
    """

    return ResNet(block_sizes = [3, 4, 6, 3],
                block_cls=ResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)


@_register_model("ResNet50")
def ResNet50(num_outputs : int, *args, **kwargs):
    """
        Build ResNet50 module.
        block_list = [3, 4, 6, 3], block_cls = BottleneckResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by BottleneckResNetBlock.
    """

    return ResNet(block_sizes = [3, 4, 6, 3],
                block_cls=BottleneckResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)


@_register_model("ResNet101")
def ResNet101(num_outputs : int, *args, **kwargs):
    """
        Build ResNet101 module.
        block_list = [3, 4, 23, 3], block_cls = BottleneckResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by BottleneckResNetBlock.
    """

    return ResNet(block_sizes = [3, 4, 23, 3],
                block_cls=BottleneckResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)

@_register_model("ResNet152")
def ResNet152(num_outputs : int, *args, **kwargs):
    """
        Build ResNet152 module.
        block_list = [3, 8, 36, 3], block_cls = BottleneckResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by BottleneckResNetBlock.
    """

    return ResNet(block_sizes = [3, 8, 36, 3],
                block_cls=BottleneckResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)


@_register_model("ResNet200")
def ResNet200(num_outputs : int, *args, **kwargs):
    """
        Build ResNet200 module.
        block_list = [3, 24, 36, 3], block_cls = BottleneckResNetBlock

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            ResNet : a ResNet Module built by BottleneckResNetBlock.
    """

    return ResNet(block_sizes = [3, 24, 36, 3],
                block_cls=BottleneckResNetBlock,
                num_outputs = num_outputs, 
                *args, **kwargs)
