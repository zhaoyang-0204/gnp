import functools
import os
import time

from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax.metrics import tensorboard
from gnp.training import utli
import jax
import jax.numpy as jnp
from tensorflow.io import gfile
from gnp.training import lr_schedule
from gnp.training import training_core

FLAGS = flags.FLAGS


def train(
			model,
			optimizer: flax.optim.Optimizer,
			state: flax.nn.Collection,
			dataset_source,
			training_dir: str, num_epochs: int):

	checkpoint_dir = os.path.join(training_dir, 'checkpoints')
	summary_writer = tensorboard.SummaryWriter(training_dir)
	if jax.host_id() != 0:  # Don't log if not first host.
		summary_writer.scalar = lambda *args: None
	prng_key = jax.random.PRNGKey(FLAGS.config.seeds)

	# Log initial results:
	if gfile.exists(checkpoint_dir):
		optimizer, state, epoch_last_checkpoint = utli.restore_checkpoint(
			optimizer, state, checkpoint_dir)
		initial_epoch = epoch_last_checkpoint + 1
		info = 'Resuming training from epoch {}'.format(initial_epoch)
		logging.info(info)
	else:
		initial_epoch = jnp.array(0, dtype=jnp.int32)
		logging.info('Starting training from scratch.')

	optimizer = jax_utils.replicate(optimizer)
	state = jax_utils.replicate(state)

	if FLAGS.config.use_learning_rate_schedule:
		if FLAGS.config.lr_schedule_type == 'cosine':
			learning_rate_fn = lr_schedule.get_cosine_schedule(num_epochs, FLAGS.config.base_lr,
													dataset_source.num_training_obs,
													dataset_source.batch_size)
		elif FLAGS.config.lr_schedule_type == 'exponential':
			learning_rate_fn = lr_schedule.get_exponential_schedule(
				num_epochs, FLAGS.config.base_lr, dataset_source.num_training_obs,
				dataset_source.batch_size)
		else:
			raise ValueError('Wrong schedule: ' + FLAGS.config.lr_schedule_type)
	else:
		learning_rate_fn = lambda step: FLAGS.config.base_lr

	# pmap the training and evaluation functions.
	pmapped_train_step = jax.pmap(
		functools.partial(
			training_core.train_step,
			learning_rate_fn=learning_rate_fn,
			l2_reg=FLAGS.config.l2_regularization,
			apply_fn = model.apply),
		axis_name='batch',
		donate_argnums=(0, 1))
	pmapped_eval_step = jax.pmap(functools.partial(training_core.eval_step, apply_fn = model.apply) , axis_name='batch')

	for epochs_id in range(initial_epoch, num_epochs):
		tick = time.time()
		optimizer, state = training_core.train_for_one_epoch(
			dataset_source, optimizer, state, prng_key, pmapped_train_step,
			summary_writer)
		tock = time.time()
		info = 'Epoch {} finished in {:.2f}s.'.format(epochs_id, tock - tick)
		logging.info(info)

		# Evaluate the model on the test set, and optionally the training set.
		if (epochs_id + 1) % FLAGS.config.evaluate_every == 0:
			info = 'Evaluating at end of epoch {} (0-indexed)'.format(epochs_id)
		logging.info(info)
		tick = time.time()
		current_step = int(optimizer.state.step[0])
		if FLAGS.config.also_eval_on_training_set:
			train_ds = dataset_source.get_train(use_augmentations=False)
			train_metrics = training_core.eval_on_dataset(
				optimizer.target, state, train_ds, pmapped_eval_step)
			for metric_name, metric_value in train_metrics.items():
				summary_writer.scalar('eval_on_train_' + metric_name,
										metric_value, current_step)
			summary_writer.flush()

		test_ds = dataset_source.get_test()
		test_metrics = training_core.eval_on_dataset(
			optimizer.target, state, test_ds, pmapped_eval_step)
		for metric_name, metric_value in test_metrics.items():
			summary_writer.scalar('test_' + metric_name,
								metric_value, current_step)
		summary_writer.flush()

		tock = time.time()
		info = 'Evaluated model in {:.2f}.'.format(tock - tick)
		logging.info(info)

		# Save new checkpoint if the last one was saved more than
		if epochs_id % FLAGS.config.save_ckpt_every_n_epochs == 0 and epochs_id != 0:
			utli.save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
			logging.info('Saved checkpoint.')
			
	# Always save final checkpoint
	utli.save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
