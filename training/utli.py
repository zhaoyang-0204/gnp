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
	Utilities.
"""

import flax
from flax.training import checkpoints
import jax.numpy as jnp
import jax
import os
from tensorflow.io import gfile
from absl import flags
from typing import Optional, Tuple

FLAGS = flags.FLAGS


def restore_checkpoint(optimizer : flax.optim.Optimizer,
                       state : flax.core.frozen_dict.FrozenDict,
                       directory : str) \
                       -> Tuple[flax.core.frozen_dict.FrozenDict,
                            flax.core.frozen_dict.FrozenDict, int]:
    """
        Restore the previous saved checkpoint in the given directory. If the
          directory contains more than one checkpoints, the latest one will be
          launched.

        Args:
            optimizer : the optimizer of the model.
            state : the state of the model.
            directory : the directory to which we will load from.

        Returns :
            A tuple that includes the resumed optimizer, state and epoch.
    """

    train_state = dict(optimizer=optimizer, model_state=state, epoch=0)
    restored_state = checkpoints.restore_checkpoint(directory, train_state)
    return (restored_state['optimizer'],
            restored_state['model_state'],
            restored_state['epoch'])


def save_checkpoint(optimizer : flax.optim.Optimizer,
                    state : flax.core.frozen_dict.FrozenDict,
                    directory : str,
                    epoch : int,
                    keep : Optional[int] = 2):
    """
        Save the checkpoint in the given directory. 

        Args:
            optimizer : the optimizer of the model.
            state : the state of the model.
            directory : the directory to which we will load from.
            epoch : the curernt training epoch.
            keep : the maximum number of checkpoints we would like to keep in
              this directory. If exceeding this maximum number when saving the
              curernt checkpoint, the ealiest checkpoint will be delete.
    """

    # Get one copy of optimizer and state from all the devices.
    optimizer = jax.tree_map(lambda x: x[0], optimizer)
    model_state = jax.tree_map(lambda x: jnp.mean(x, axis=0), state)
    train_state = dict(optimizer=optimizer, model_state=model_state, epoch=epoch)
    if gfile.exists(os.path.join(directory, 'checkpoint_' + str(epoch))):
        gfile.remove(os.path.join(directory, 'checkpoint_' + str(epoch)))
    checkpoints.save_checkpoint(directory, train_state, epoch, keep=keep)


def cross_entropy_loss(logits : jnp.ndarray,
                       one_hot_labels : jnp.ndarray,
                       mask : Optional[jnp.ndarray] = None):
    """
        Compute the cross entropy loss between the predicted label (logits) and
          the true label (one_hot_labels). 

        Args:
            logits :  the predicted logits by the model.
            one_hot_labels : the true label for this logits.
            mask : the array that indicates the validaity for this batch.

        Returns:
            loss : the cross entropy loss.
    """

    # Apply label smoothing if needed.
    if FLAGS.config.label_smoothing > 0:
        smoothing = jnp.ones_like(one_hot_labels) / one_hot_labels.shape[-1]
        one_hot_labels = ((1-FLAGS.config.label_smoothing) * one_hot_labels
                        + FLAGS.config.label_smoothing * smoothing)
    # Apply the softmax to the logits.
    log_softmax_logits = jax.nn.log_softmax(logits)
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0], 1])
    # Compute the cross entropy loss.
    loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
    return jnp.nan_to_num(loss)


def error_rate_metric(logits : jnp.ndarray,
                      one_hot_labels : jnp.ndarray,
                      mask : Optional[jnp.ndarray] = None):
    """
        Compute the error rate between the predicted label (logits) and the true
          label (one_hot_labels). 

        Args:
            logits :  the predicted logits by the model.
            one_hot_labels : the true label for this logits.
            mask : the array that indicates the validaity for this batch.

        Returns:
            error_rate : the error rate.
    """

    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    # Count the number of valid samples.
    mask = mask.reshape([logits.shape[0]])
    # Compute the error rate, i.e. wrong predictions / total number of valid
    # samples.
    error_rate = (((jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1))) *
                    mask).sum() / mask.sum()
    return jnp.nan_to_num(error_rate)


def top_k_error_rate_metric(logits : jnp.ndarray,
                            one_hot_labels : jnp.ndarray,
                            k : Optional[int] = 5,
                            mask : Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
        Compute the top k error rate between the predicted label (logits) and
        the true label (one_hot_labels). 

        Args:
            logits :  the predicted logits by the model.
            one_hot_labels : the true label for this logits.
            k : the top K error rate to compute.
            mask : the array that indicates the validaity for this batch.

        Returns:
            error_rate : the top K error rate.
    """

    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - ((hit * mask).sum() / mask.sum())
    return jnp.nan_to_num(error_rate)


def tensorflow_to_numpy(xs) -> jnp.ndarray:
    """
        Convert tf tensors to numpy.

        Args:
            xs : a pytree (such as nested tuples, lists, and dicts) where the
              leaves are tensorflow tensors.
        Returns:
            xs : a pytree with the same structure as xs, where the leaves have
              been converted to jax numpy ndarrays.
    """

    xs = jax.tree_map(lambda x: x._numpy(), xs) 
    return xs

def shard_batch(xs : jnp.ndarray) -> jnp.ndarray:
    """
        Shard a given batch to align to all devices.

        This actually will reshape the original batch data from shape 
          [batch, size, size, channels] to [num_devices, batch // num_devices,
          size, size, channels], and then distribute the reshaped batch to each
          device based on the first dimension of the reshpaed batch. So, error
          will be raised if the batch_size is not divisible by the number of
          devices.

        Args:
            xs : a pytree with the same structure as xs, where the leaves have
              been converted to jax numpy ndarrays.
        Returns:
            a pytree with the same structure as xs, where the first dimension of
              xs is reshaped to align to the number of devices for distribution.
    """

    local_device_count = jax.local_device_count()
    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.tree_map(_prepare, xs)


def global_norm(x : flax.core.frozen_dict.FrozenDict) -> flax.core.frozen_dict.FrozenDict:
    """
        Compute the norm of a given input with pytree structure.

        Args:
            x : a pytree data.
        Returns:
            gn : a pytree with the same structure as xs, where the leaves are
                  reshaped over the first dimension to align to the number of
                  devices for distribution.
    """ 

    gn =  jnp.sqrt(sum([jnp.sum(jnp.square(leaf)) for leaf in jax.tree_leaves(x)]))
    return gn


def clip_by_global_norm(x : flax.core.frozen_dict.FrozenDict) \
     -> flax.core.frozen_dict.FrozenDict:
    """
        Clip the given data to norm.

        Args:
            x : a pytree data.
        Returns:
            x_clipped : the data that has been clipped to its norm.
    """ 

    if FLAGS.config.gradient_clipping > 0:
        g_norm = global_norm(x)
        trigger = g_norm < FLAGS.config.gradient_clipping
        x_clipped = jax.tree_multimap(
            lambda t: jnp.where(trigger, t, (t / g_norm) * FLAGS.config.gradient_clipping),
            x)
    return x_clipped


def dual_vector(x: flax.core.frozen_dict.FrozenDict) \
      -> flax.core.frozen_dict.FrozenDict:
    """
        Compute the dual vector for a given pytree data. Given x, the dual
          vector of ||x||_2 is that x / ||x||_2.

        Args:
            x : a pytree data.
        Returns:
            normalized_gradient : the pytree data with all its elements
              normalized.
    """ 

    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(x)]))
    normalized_gradient = jax.tree_map(lambda t: t / gradient_norm, x)
    return normalized_gradient


def asam_vector(w : flax.core.frozen_dict.FrozenDict,
                g : flax.core.frozen_dict.FrozenDict) \
                    -> flax.core.frozen_dict.FrozenDict:
    """
        Compute the asam vector for a given pytree data. See Adaptive SAM paper
          for details. If you would like to try ASAM, you should add this in the 
          get_gnp_gradient function.

        Args:
            w : the weight of the model.
            g : the current gradient.
        Returns:
            asam_gradient : the asam gradient.
    """ 
    w_abs = jax.tree_map(lambda x: jnp.abs(x), w)
    Tw_g = jax.tree_multimap(lambda a, b: a * b, w_abs, g)
    Tw2_g = jax.tree_multimap(lambda a, b: a * a * b, w_abs, g)
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(Tw_g)]))
    asam_gradient = jax.tree_map(lambda x: x / gradient_norm, Tw2_g)
    return asam_gradient