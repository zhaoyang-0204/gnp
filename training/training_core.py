import flax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
from gnp.training import utli
from absl import flags
import jax
from absl import logging
from flax.training import common_utils
import time

FLAGS = flags.FLAGS

def train_step(
        optimizer: flax.optim.Optimizer,
        state: flax.nn.Collection,
        batch: Dict[str, jnp.ndarray],
        prng_key: jnp.ndarray,
        learning_rate_fn: Callable[[int], float],
        l2_reg: float,
        apply_fn : Callable,
    ):
    def forward_and_loss(params: flax.core.frozen_dict.FrozenDict,
                        #  state: flax.core.frozen_dict.FrozenDict,
                        true_gradient = False):
        logits, new_state = apply_fn(
            variables = {"params":params, **state},
            inputs = batch['image'],
            train = True,
            rngs = dict(dropout = prng_key, shake = prng_key),
            mutable = list(state.keys()),
            true_gradient = true_gradient
        )
        loss = utli.cross_entropy_loss(logits, batch['label'])
        # We apply weight decay to all parameters, including bias and batch norm
        # parameters.
        weight_penalty_params = jax.tree_leaves(params)
        if FLAGS.config.no_weight_decay_on_bn:
            weight_l2 = sum(
                [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        else:
            weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])

        weight_penalty = l2_reg * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, (new_state, logits)

    step = optimizer.state.step

    def get_gnp_gradient(model: flax.nn.Model, *, r: float, alpha : float):
        # compute gradient on the whole batch
        (_, (inner_state, _)), grad = jax.value_and_grad(
            lambda m: forward_and_loss(m, true_gradient = True), has_aux=True)(model)
        if FLAGS.config.gnp.sync_perturbations:
            grad = jax.lax.pmean(grad, 'batch')

        grad_origial = grad
        if FLAGS.config.asam:
            logging.info("Using ASAM ...")
            grad = utli.asam_vector(model, grad)
        else:
            grad = utli.dual_vector(grad)
        noised_model = jax.tree_multimap(lambda a, b: a + r * b,
                                        model, grad)
        (_, (_, logits)), grad_noised = jax.value_and_grad(forward_and_loss, has_aux=True)(noised_model)
        if FLAGS.config.gnp.norm_perturbations:
            if FLAGS.config.use_dual_in_adam:
                g = jax.tree_multimap(lambda a, b: (1 - alpha) * a + alpha * b, utli.dual_vector(grad), grad_noised)
            else:
                logging.info("Using Norm gradient!")
                g = jax.tree_multimap(lambda a, b: (1 - alpha) * a + alpha * b, grad, grad_noised)
        else:
            logging.info("Using Unnormalized gradient!")
            g = jax.tree_multimap(lambda a, b: (1 - alpha) * a + alpha * b, grad_origial, grad_noised)
        return (inner_state, logits), g

    lr = learning_rate_fn(step)
    r_value = FLAGS.config.gnp.r
    alpha = FLAGS.config.gnp.alpha

    if r_value > 0:  # SAM loss
        (new_state, logits), grad = get_gnp_gradient(optimizer.target, r = r_value, alpha = alpha)
    else:  # Standard SGD
        (_, (new_state, logits)), grad = jax.value_and_grad(
            forward_and_loss, has_aux=True)(
                optimizer.target)

    # We synchronize the gradients across replicas by averaging them.
    grad = jax.lax.pmean(grad, 'batch')

    # Gradient is clipped after being synchronized.
    grad = utli.clip_by_global_norm(grad)
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    # Compute some norms to log on tensorboard.
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)]))
    param_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(
            new_optimizer.target)]))

    # Compute some metrics to monitor the training.
    metrics = {'train_error_rate': utli.error_rate_metric(logits, batch['label']),
                'train_loss': utli.cross_entropy_loss(logits, batch['label']),
                'gradient_norm': gradient_norm,
                'param_norm': param_norm}

    return new_optimizer, new_state, metrics, lr

        
def eval_step(params: flax.nn.Model, state: flax.nn.Collection,
                batch: Dict[str, jnp.ndarray],
                apply_fn) -> Dict[str, float]:

    # Averages the batch norm moving averages.
    state = jax.lax.pmean(state, 'batch')
    logits = apply_fn(
        variables = {"params" : params, **state},
        train = False,
        mutable = False,
        inputs = batch['image']
    )
    num_samples = (batch['image'].shape[0] if 'mask' not in batch
                    else batch['mask'].sum())
    mask = batch.get('mask', None)
    labels = batch['label']
    metrics = {
        'error_rate':
            utli.error_rate_metric(logits, labels, mask) * num_samples,
        'loss':
            utli.cross_entropy_loss(logits, labels, mask) * num_samples
    }
    if FLAGS.config.compute_top_5_error_rate:
        metrics.update({
            'top_5_error_rate':
                utli.top_k_error_rate_metric(logits, labels, 5, mask) * num_samples
        })
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics

def eval_on_dataset(
        model: flax.nn.Model, state: flax.nn.Collection, dataset,
        pmapped_eval_step
    ):

    eval_metrics = []
    total_num_samples = 0
    all_host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')

    for eval_batch in dataset:
        # Load and shard the TF batch.
        eval_batch = utli.load_and_shard_tf_batch(eval_batch)
        # Compute metrics and sum over all observations in the batch.
        metrics = pmapped_eval_step(model, state, eval_batch)
        eval_metrics.append(metrics)
        if 'mask' not in eval_batch:
            total_num_samples += (
                eval_batch['label'].shape[0] * eval_batch['label'].shape[1] *
                jax.host_count())
        else:
            total_num_samples += all_host_psum(eval_batch['mask'])[0].sum()

    # Metrics are all the same across all replicas (since we applied psum in the
    # eval_step). The next line will fetch the metrics on one of them.
    eval_metrics = common_utils.get_metrics(eval_metrics)
    # Finally, we divide by the number of samples to get the mean error rate and
    # cross entropy.
    eval_summary = jax.tree_map(lambda x: x.sum() / total_num_samples,
                                eval_metrics)
    print(eval_summary)
    return eval_summary


def train_for_one_epoch(
        dataset_source,
        optimizer: flax.optim.Optimizer, state: flax.nn.Collection,
        prng_key: jnp.ndarray, pmapped_train_step,
        summary_writer
    ):

    start_time = time.time()
    cnt = 0
    train_metrics = []
    for batch in dataset_source.get_train(use_augmentations=True):
        # Generate a PRNG key that will be rolled into the batch.
        step_key = jax.random.fold_in(prng_key, optimizer.state.step[0])
        # Load and shard the TF batch.
        batch = utli.tensorflow_to_numpy(batch)
        batch = utli.shard_batch(batch)
        # Shard the step PRNG key.
        sharded_keys = common_utils.shard_prng_key(step_key)

        optimizer, state, metrics, lr = pmapped_train_step(
            optimizer, state, batch, sharded_keys)
        cnt += 1

        if cnt % 500 == 0 and cnt != 0:
            print("Steps : %d"%cnt, f"Time Cost : {time.time() - start_time}", metrics)

        train_metrics.append(metrics)
    train_metrics = common_utils.get_metrics(train_metrics)
    # Get training epoch summary for logging.
    train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
    train_summary['learning_rate'] = lr[0]
    current_step = int(optimizer.state.step[0])
    info = 'Whole training step done in {} ({} steps)'.format(
        time.time()-start_time, cnt)
    logging.info(info)
    for metric_name, metric_value in train_summary.items():
        summary_writer.scalar(metric_name, metric_value, current_step)
    summary_writer.flush()
    return optimizer, state
