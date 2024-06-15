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
    Main program interface.
"""

from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import jax
import tensorflow as tf
from gnp.main import util
from gnp.models import load_model
from tensorflow.io import gfile
from gnp.training import training
from gnp.ds_pipeline.get_dataset import get_dataset_pipeline
from gnp.optimizer.get_optimizer import get_optimizer, manage_optimizer_config
from gnp.training import training


FLAGS = flags.FLAGS
WORK_DIR = flags.DEFINE_string('working_dir', None,
                               'Directory to store configs, logs and checkpoints.')
config_flags.DEFINE_config_file(
    "config",
    None,
    'File path to the training hyperparameter configuration.'
)
flags.mark_flags_as_required(["config", "working_dir"])


def main(_):

    # Avoids tf allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    
    ### This follows the SAM repo. If using TPUs, uncomment set_hardware_bernoulli().  
    # Performance gains on TPU by switching to hardware bernoulli.
    def hardware_bernoulli(rng_key, p=jax.numpy.float32(0.5), shape=None):
        lax_key = jax.lax.tie_in(rng_key, 0.0)
        return jax.lax.rng_uniform(lax_key, 1.0, shape) < p
    def set_hardware_bernoulli():
        jax.random.bernoulli = hardware_bernoulli
    # set_hardware_bernoulli()

    # Adjust the optimizer config flags based on the optimizer type.
    manage_optimizer_config()
    
    # Create working folder for the training.
    model_folder = util.create_model_folder()
    FLAGS.config.model_folder = model_folder

    # Get dataset pipeline.
    ds = get_dataset_pipeline()

    # Get model instance.
    model, variables = load_model.get_model(FLAGS.config.model.model_name,
                                                FLAGS.config.batch_size,
                                                FLAGS.config.dataset.image_size,
                                                FLAGS.config.dataset.num_classes,
                                                FLAGS.config.dataset.num_channels,
                                                FLAGS.config.init_seeds)

    # Get optimizer instance. 
    optimizer = get_optimizer()

    # Start training.
    training.train(model,
                   optimizer,
                   variables,
                   ds,
                   FLAGS.config.model_folder,
                   FLAGS.config.total_epochs)
                        

if __name__ == '__main__':
  app.run(main)
