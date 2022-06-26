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
    Utilities. Some initialization methods.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import flax.linen as nn
from functools import partial

# Batch norm parameters.
_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5


# Kaiming initialization with fan out mode. Should be used to initialize
# convolutional kernels.
conv_kernel_init = jax.nn.initializers.variance_scaling(2.0, "fan_out", "normal")


def dense_layer_init_fn(key: jnp.ndarray,
                        shape: Tuple[int, int],
                        dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Initializer for the final dense layer.

    Forked from:
    https://github.com/google-research/sam.

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
    def __call__(self,
                 x : jnp.ndarray,
                 train : bool,
                 apply_relu : Optional[bool] = True):

        """
            Activition operation layer, which combines a batch norm layer and a
              optional added relu activition layer.

            Args:
                x: inputs to the layer.
                train : train flag. Decide whether to use moving average in BN
                  layer or not.
                apply_relu : if ture, apply relu activation after batch norm.

            Returns:
                x : output of the layer.
        """
        
        norm = partial(nn.BatchNorm, use_running_average = not train,
                            momentum = _BATCHNORM_MOMENTUM, epsilon = _BATCHNORM_EPSILON,
                            ) 
        x = norm()(x)
        if apply_relu: x = jax.nn.relu(x)
        return x