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
    Vision Transformer Model. 

    Reference:
      An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
      [https://arxiv.org/abs/2010.11929]

    This implementation mimics that in 
    https://github.com/google-research/vision_transformer
"""

from flax import linen as nn
from jax import numpy as jnp
from typing import Any, Callable, Optional, Tuple
import ml_collections
from gnp.models import _register_model


Array = Any
PRNGKey = Any
Shape = Tuple[int]

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self,
                 x : jnp.ndarray):
        """
            Apply Transformer Encoder module.

            Args:
                inputs : inputs to the module.


            Returns:
                outputs of the module.
        """
        return x


class AddPositionEmbs(nn.Module):
    """
        Add positional embeddings to the inputs.

        Attributes:
            posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape], Array]

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray):
        """
            Apply Positional Embedding module.

            Args:
                inputs : inputs to the module.

            Returns:
                outputs of the module.
        """

        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pe



class MlpBlock(nn.Module):
    """
        Mlp Block Module.

        Attributes:
            mlp_dim : the dimension of first mlp layer.
            out_dim : the dimension of output layer.
            dropout_rate : the dropout rate in dense layers.
            kernel_init : the kernel initializer.
            bias_init : the bias initializer.
    """

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 deterministic : bool):
        """
            Apply Mlp Block module.

            Args:
                inputs : inputs to the module.
                deterministic : deterministic flag. Decide whether to mask neurons
                  in layer by dropout or not.

            Returns:
                x : outputs of the module.
        """

        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(inputs)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        output = nn.Dense(features=actual_out_dim,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)
        
        output = nn.Dropout(rate = self.dropout_rate)(output, deterministic=deterministic)

        return output



class TransformerBlock(nn.Module):
    """
        Transformer Block Module.

        Attributes:
            mlp_dim : the dimension of mlp layer.
            num_heads : the number of attention head.
            dropout_rate : the dropout rate in dense layers.
            attention_dropout_rate : the dropout rate for attentions.
    """

    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 deterministic : bool):
        """
            Apply Transformer Encoder module.

            Args:
                inputs : inputs to the module.
                deterministic : deterministic flag. Decide whether to mask neurons
                  in layer by dropout or not.

            Returns:
                x : outputs of the module.
        """

        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
                                            broadcast_dropout=False,
                                            deterministic=deterministic,
                                            dropout_rate = self.attention_dropout_rate,
                                            num_heads = self.num_heads)(x, x)
        
        x = nn.Dropout(self.dropout_rate)(x, deterministic = deterministic)
        x = x + inputs

        y = nn.LayerNorm()(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """
        Transformer Encoder Module.

        Attributes:
            num_layers : the number of transformer block.
            mlp_dim : the dimension of mlp layer.
            num_heads : the number of attention head.
            dropout_rate : the dropout rate in dense layers.
            attention_dropout_rate : the dropout rate for attentions.
        
        Returns:
            x : output.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool):
        """
            Apply Transformer Encoder module.

            Args:
                inputs : inputs to the module.
                train : train flag. 

            Returns:
                x : outputs of the module.
        """

        x = AddPositionEmbs(
            posemb_init = nn.initializers.normal(stddev=0.02),
            name = "posembed_input"
        )(inputs)
        x = nn.Dropout(self.dropout_rate, deterministic= not train)(x)

        for lyr in range(self.num_layers):
            x = TransformerBlock(
                mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name = f"encoderblock_{lyr}",
                num_heads=self.num_heads
            )(x, deterministic=not train)

        encoded = nn.LayerNorm(name = "encoder_norm")(x)

        return encoded


@_register_model("ViTBase")
class VisionTransformer(nn.Module):
    """
        Vision Transformer Module.

        Attributes:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.
            patches : a config dict with size property, which represents the
              size of the patches the input images will be cut into.
            transformer : a config dict that specifies the parameters in the
              transformer encoder block.
            hidden_size : the hidden size in the first convolutional layer,
             i.e. the dimension of the features (channels).
            representation_size : if not None (default is None), an additional
              dense layer will be added before the output layer.
        
        Returns:
            x : output.
    """

    num_outputs: int
    patches : ml_collections.ConfigDict
    transformer:  ml_collections.ConfigDict
    hidden_size: int
    representation_size: Optional[int] = None

    @nn.compact
    def __call__(self,
                 inputs : jnp.ndarray,
                 train : bool):
        """
            Apply Vision Transformer module.

            Args:
                inputs : inputs to the module.
                train : train flag. 

            Returns:
                x : outputs of the module.
        """

        x = inputs
        # Convolution the input to 
        x = nn.Conv(self.hidden_size, kernel_size=self.patches.size,
                    strides=self.patches.size, padding = "VALID", name = "embedding")(x)

        # Reshape 2D feature to 1D.
        n, h, w, embedding_dim = x.shape
        x =  jnp.reshape(x, [n, h * w, embedding_dim])

        # Add cls tocken.
        cls = self.param("cls", nn.initializers.zeros, (1, 1, embedding_dim))
        cls = jnp.tile(cls, [n, 1, 1])        
        x = jnp.concatenate([cls, x], axis = 1)

        # Apply transformer encoder module
        x = Encoder(name = "Transformer", **self.transformer)(x, train = train)

        # Extract the cls tocken as feature representations for output.
        x = x[:, 0]

        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name = "pre_logits")(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name='pre_logits')(x)
        
        if self.num_outputs:
            x = nn.Dense(
                features=self.num_outputs,
                name = "head",
                kernel_init=nn.initializers.zeros
            )(x)
    
        return x


@_register_model("ViT_TI16")
def VIT_TI16(num_outputs : int,
             patches : Optional[ml_collections.ConfigDict] = None,
             *args,
             **kwargs):
    """
        Build ViT-Tiny-16 module. We provide two basic modules for training on
          Cifar and ImageNet dataset. The patch size is (4, 4) for Cifar dataset
          while it is (16, 16) for ImageNet dataset. For other patch size, you
          could pass it to patches arg.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VisionTransformer : a VisionTransformer Module.
    """

    if patches is None:
        if num_outputs in (10, 100):
            patches=ml_collections.ConfigDict(dict(size=(4, 4)))
        elif num_outputs == 1000:
            patches=ml_collections.ConfigDict(dict(size=(16, 16)))
        else:
            assert patches is not None

    return VisionTransformer(
        num_outputs=num_outputs,
        patches=patches,
        hidden_size=192,
        transformer= ml_collections.ConfigDict(
                dict(mlp_dim = 768, num_heads = 3, num_layers = 12)
            ),
        representation_size = None, *args, **kwargs
    )


@_register_model("ViT_S16")
def VIT_S16(num_outputs : int,
             patches : Optional[ml_collections.ConfigDict] = None,
             *args,
             **kwargs):
    """
        Build ViT-Small-16 module. We provide two basic modules for training on
          Cifar and ImageNet dataset. The patch size is (4, 4) for Cifar dataset
          while it is (16, 16) for ImageNet dataset. For other patch size, you
          could pass it to patches arg.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VisionTransformer : a VisionTransformer Module.
    """

    if num_outputs in (10, 100):
        patches=ml_collections.ConfigDict(dict(size=(4, 4)))
    elif num_outputs == 1000:
        patches=ml_collections.ConfigDict(dict(size=(16, 16)))
    else:
        assert patches is not None

    return VisionTransformer(
        num_outputs=num_outputs,
        patches=patches,
        hidden_size=384,
        transformer= ml_collections.ConfigDict(
                dict(mlp_dim = 1536, num_heads = 6, num_layers = 12)
            ),
        representation_size = None, *args, **kwargs
    )


@_register_model("ViT_B16")
def VIT_B16(num_outputs : int,
             patches : Optional[ml_collections.ConfigDict] = None,
             *args,
             **kwargs):
    """
        Build ViT-Basic-16 module. We provide two basic modules for training on
          Cifar and ImageNet dataset. The patch size is (4, 4) for Cifar dataset
          while it is (16, 16) for ImageNet dataset. For other patch size, you
          could pass it to patches arg.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VisionTransformer : a VisionTransformer Module.
    """

    if num_outputs in (10, 100):
        patches=ml_collections.ConfigDict(dict(size=(4, 4)))
    elif num_outputs == 1000:
        patches=ml_collections.ConfigDict(dict(size=(16, 16)))
    else:
        assert patches is not None
    
    return VisionTransformer(
        num_outputs=num_outputs,
        patches=patches,
        hidden_size=768,
        transformer= ml_collections.ConfigDict(
                dict(mlp_dim = 3072, num_heads = 12, num_layers = 12)
            ),
        representation_size = None, *args, **kwargs
    )


@_register_model("ViT_L16")
def VIT_L16(num_outputs : int,
             patches : Optional[ml_collections.ConfigDict] = None,
             *args,
             **kwargs):
    """
        Build ViT-Large-16 module. We provide two basic modules for training on
          Cifar and ImageNet dataset. The patch size is (4, 4) for Cifar dataset
          while it is (16, 16) for ImageNet dataset. For other patch size, you
          could pass it to patches arg.

        Args:
            num_outputs : the dimension of outputs, i.e. the number of neurons
              in the output dense layer.

        Returns:
            VisionTransformer : a VisionTransformer Module.
    """
    
    if num_outputs in (10, 100):
        patches=ml_collections.ConfigDict(dict(size=(4, 4)))
    elif num_outputs == 1000:
        patches=ml_collections.ConfigDict(dict(size=(16, 16)))
    else:
        assert patches is not None
    
    return VisionTransformer(
        num_outputs=num_outputs,
        patches=patches,
        hidden_size=1024,
        transformer= ml_collections.ConfigDict(
                dict(mlp_dim = 4096, num_heads = 16, num_layers = 24)
            ),
        representation_size = None, *args, **kwargs
    )

