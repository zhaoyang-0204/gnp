import flax
from flax.training import checkpoints
import jax.numpy as jnp
import jax
import os
from tensorflow.io import gfile
from flax import optim
from absl import flags

FLAGS = flags.FLAGS


def create_optimizer(model: flax.nn.Model,
                     learning_rate: float,
                     beta: float = 0.9) -> flax.optim.Optimizer:
    """Creates an optimizer.

    Learning rate will be ignored when using a learning rate schedule.

    Args:
        model: The FLAX model to optimize.
        learning_rate: Learning rate for the gradient descent.
        beta: Momentum parameter.

    Returns:
        A SGD (or RMSProp) optimizer that targets the model.
    """
    optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                beta=beta,
                                nesterov=True)
    optimizer = optimizer_def.create(model)
    return optimizer


def restore_checkpoint(
        optimizer: flax.optim.Optimizer,
        model_state,
        directory):

    train_state = dict(optimizer=optimizer, model_state=model_state, epoch=0)
    restored_state = checkpoints.restore_checkpoint(directory, train_state)
    return (restored_state['optimizer'],
            restored_state['model_state'],
            restored_state['epoch'])


def save_checkpoint(optimizer: flax.optim.Optimizer,
                    model_state,
                    directory: str,
                    epoch: int):

    optimizer = jax.tree_map(lambda x: x[0], optimizer)
    model_state = jax.tree_map(lambda x: jnp.mean(x, axis=0), model_state)
    train_state = dict(optimizer=optimizer,
                        model_state=model_state,
                        epoch=epoch)
    if gfile.exists(os.path.join(directory, 'checkpoint_' + str(epoch))):
        gfile.remove(os.path.join(directory, 'checkpoint_' + str(epoch)))
    checkpoints.save_checkpoint(directory, train_state, epoch, keep=2)


def cross_entropy_loss(logits: jnp.ndarray,
                       one_hot_labels: jnp.ndarray,
                       mask = None):

    if FLAGS.config.label_smoothing > 0:
        smoothing = jnp.ones_like(one_hot_labels) / one_hot_labels.shape[-1]
        one_hot_labels = ((1-FLAGS.config.label_smoothing) * one_hot_labels
                        + FLAGS.config.label_smoothing * smoothing)
    log_softmax_logits = jax.nn.log_softmax(logits)
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0], 1])
    loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
    return jnp.nan_to_num(loss)


def error_rate_metric(logits: jnp.ndarray,
                      one_hot_labels: jnp.ndarray,
                      mask = None):

    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    error_rate = (((jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1))) *
                    mask).sum() / mask.sum()
    return jnp.nan_to_num(error_rate)


def top_k_error_rate_metric(logits: jnp.ndarray,
                            one_hot_labels: jnp.ndarray,
                            k: int = 5,
                            mask = None) -> jnp.ndarray:

    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - ((hit * mask).sum() / mask.sum())
    return jnp.nan_to_num(error_rate)

def tensorflow_to_numpy(xs):
    xs = jax.tree_map(lambda x: x._numpy(), xs) 
    return xs

def shard_batch(xs):
    local_device_count = jax.local_device_count()
    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.tree_map(_prepare, xs)

def load_and_shard_tf_batch(xs):
    return shard_batch(tensorflow_to_numpy(xs))

def global_norm(updates):
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def clip_by_global_norm(updates):
    if FLAGS.config.gradient_clipping > 0:
        g_norm = global_norm(updates)
        trigger = g_norm < FLAGS.config.gradient_clipping
        updates = jax.tree_multimap(
            lambda t: jnp.where(trigger, t, (t / g_norm) * FLAGS.config.gradient_clipping),
            updates)
    return updates


def dual_vector(y: jnp.ndarray):
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
    normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
    return normalized_gradient


def asam_vector(w, g: jnp.ndarray):
    w_abs = jax.tree_map(lambda x: jnp.abs(x), w)
    Tw_g = jax.tree_multimap(lambda a, b: a * b, w_abs, g)
    Tw2_g = jax.tree_multimap(lambda a, b: a * a * b, w_abs, g)
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(Tw_g)]))
    asam_gradient = jax.tree_map(lambda x: x / gradient_norm, Tw2_g)
    return asam_gradient