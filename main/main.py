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

from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import jax
import flax
import tensorflow as tf

from gnp.main import utli
from gnp.models import load_model
from tensorflow.io import gfile
from gnp.training import training
from gnp.ds_pipeline.get_dataset import get_dataset_pipeline
from gnp.optimizer.get_optimizer import get_optimizer
from gnp.training import training

FLAGS = flags.FLAGS
WORK_DIR = flags.DEFINE_string('working_dir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    "config",
    None,
    'File path to the training hyperparameter configuration.'
)
flags.mark_flags_as_required(["config", "working_dir"])


def main(_):

    tf.config.experimental.set_visible_devices([], 'GPU')
    
    # Performance gains on TPU by switching to hardware bernoulli.
    def hardware_bernoulli(rng_key, p=jax.numpy.float32(0.5), shape=None):
        lax_key = jax.lax.tie_in(rng_key, 0.0)
        return jax.lax.rng_uniform(lax_key, 1.0, shape) < p

    def set_hardware_bernoulli():
        jax.random.bernoulli = hardware_bernoulli

    # set_hardware_bernoulli()

    model_folder = utli.create_model_folder()
    FLAGS.config.model_folder = model_folder
    batch_size = FLAGS.config.batch_size 

    ds = get_dataset_pipeline()

    module, params, state = load_model.get_model(FLAGS.config.model.model_name,
                                        batch_size, FLAGS.config.dataset.image_size,
                                        FLAGS.config.dataset.num_classes,
                                        FLAGS.config.dataset.num_channels,
                                        FLAGS.config.init_seeds)

    optimizer = get_optimizer(params, 0.0)

    training.train(module, optimizer, state, ds, FLAGS.config.model_folder,
                        FLAGS.config.total_epochs)

if __name__ == '__main__':
  app.run(main)
