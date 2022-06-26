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
    Get dataset pipeline.

    The dataset pipeline class are forked from the SAM repo,
      https://github.com/google-research/sam.

    Note that when loading ImageNet dataset, some methods in this repo is
      rewritten to use the local data, instead of the downloaded
      tensorflow_datasets like SAM. One should specify the path to the local
      dataset folders in the basic config flags. If you would like to change the
      flags defined in this folder, you could specify it when executing, e.g.
      python ... --randaug_num_layers=4
"""

from absl import flags
from gnp.ds_pipeline.datasets import dataset_source, dataset_source_imagenet

FLAGS = flags.FLAGS


def get_dataset_pipeline():

    """
        Get dataset pipeline based on the dataset_name in config.

        For currently, the available dataset are 
          __AVAILABLE_DATASET = {
              "cifar10", "cifar100", "imagenet"
          }

    """

    if FLAGS.config.dataset.dataset_name == 'cifar10':
        image_size = None
        ds = dataset_source.Cifar10(
            FLAGS.config.batch_size,
            FLAGS.config.dataset.image_level_augmentations,
            FLAGS.config.dataset.batch_level_augmentations,
            image_size=image_size
        )
    elif FLAGS.config.dataset.dataset_name == 'cifar100':
        image_size = None
        ds = dataset_source.Cifar100(
            FLAGS.config.batch_size, FLAGS.config.dataset.image_level_augmentations,
            FLAGS.config.dataset.batch_level_augmentations, image_size=image_size
        )
    elif FLAGS.config.dataset.dataset_name == 'imagenet':
        image_size = 224
        ds = dataset_source_imagenet.Imagenet(
            batch_size = FLAGS.config.batch_size, image_level_augmentations = FLAGS.config.dataset.image_level_augmentations,
            image_size=image_size
        )
        FLAGS.config.dataset.image_size = image_size
    else:
        raise ValueError('Dataset not recognized.')

    if FLAGS.config.dataset.dataset_name == 'cifar10' or FLAGS.config.dataset.dataset_name == 'cifar100':
        if image_size is None:
            FLAGS.config.dataset.image_size = 32
            FLAGS.config.dataset.num_channels = 3
            FLAGS.config.dataset.num_classes = 10 if FLAGS.config.dataset.dataset_name == 'cifar10' else 100
    elif FLAGS.config.dataset.dataset_name == 'imagenet':   
        FLAGS.config.dataset.num_channels = 3
        FLAGS.config.dataset.num_classes = 1000
    else:
        raise ValueError('Dataset not recognized.')

    return ds

