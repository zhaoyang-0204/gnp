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
	Build flax models for image classification. All models are in the
	available_models folder. For now, avaiable models are
	{
		"VGG Family" : ("VGG16", "VGG19", "VGG16BN", "VGG19BN"),
	    "ResNet Family" : ("ResNet18", "ResNet34", "ResNet50", "ResNet101",
	    				   "ResNet152", "ResNet200"),
		"WideResNet Family" : ("WideResNet_16_4", "WideResNet_28_10",
							   "WideResNet_40_10", "WideResNet_2_96_ShakeShake"),
		"PyramidNet Family" : ("PyramidNet_164_48", "PyramidNet_164_270",
                               "PyramidNet_200_240", "PyramidNet_272_200",
                               "PyramidNet_164_48_ShakeDrop",
                               "PyramidNet_272_200_ShakeDrop")
        "ViT Family" : ("ViT_TI16", "ViT_S16", "ViT_B16", "ViT_L16")
	}

"""

from typing import Optional, Tuple
import flax
from jax import numpy as jnp
from jax import random
from gnp.models import MODEL_SET
import gnp.models.available_models
from absl import logging, flags

FLAGS = flags.FLAGS


def init_image_model(
	prng_key: jnp.ndarray,
	batch_size: int,
	image_size: int,
	module: flax.linen.Module,
	num_channels: Optional[int] = 3) \
		-> Tuple[flax.core.frozen_dict.FrozenDict, flax.core.frozen_dict.FrozenDict]:
	"""
		Instantiates a Flax model. 

		Args:
			prng_key : PRNG key used for initializing model weights.
			batch_size : batch size. 
			image_size : image size.
			module : flax module to be instantiated.
			num_channels : channels of the input.
		Returns:
		    params : the initialized parameters of the model after model.init().
			state : the initialized state of the model after model.init().
	"""
	dummy_input = jnp.zeros(shape = (batch_size, image_size, image_size, num_channels))
	variables = module.init(
		prng_key, dummy_input, train = False
	)
	state, params = variables.pop("params")
	return params, state


def get_model(
	model_name: str,
	batch_size: int,
	image_size: int,
	num_classes: int,
	num_channels: int = 3,
	init_seeds: Optional[int] = None,
	) -> \
		Tuple[flax.linen.Module, flax.core.frozen_dict.FrozenDict, flax.core.frozen_dict.FrozenDict]:
	"""
		Get a Flax model instance, together with its parameters and states. 

		Args:
			model_name : name of the model to be instantiated.
			batch_size : batch size. 
			image_size : image size.
		    num_classes : number of classes in the datasets, i.e. the dimension
		      of output layer.
			num_channels : channels of the input.
			init_seeds : seeds of PRNG key used for initializing model weights.
		Returns:
			model : flax module instance.
		    params : the initialized parameters of the model after model.init().
			state : the initialized state of the model after model.init().
	"""	

	logging.info(f"Available Models : {MODEL_SET.keys()}")
	assert model_name in MODEL_SET.keys()
	model = MODEL_SET[model_name](num_outputs=num_classes)
	
	FLAGS.config.unlock()
	FLAGS.config.has_true_gradient = ("Shake" in model_name)
	FLAGS.config.lock()
	if FLAGS.config.has_true_gradient:
		logging.info("True gradient will be applied during training for shake regularization in GNP.")

	if not init_seeds:
		prng_key = random.PRNGKey(init_seeds)
	else:
		prng_key = random.PRNGKey(init_seeds)
	
	params, init_state = init_image_model(
		prng_key, batch_size, image_size, model, num_channels
	)

	return model, params, init_state
