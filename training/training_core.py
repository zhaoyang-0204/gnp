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
	The core functions for training.
"""

import flax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
from gnp.training import utli
from absl import flags
import jax
from flax.training import common_utils
from gnp.ds_pipeline.get_dataset import dataset_source

FLAGS = flags.FLAGS

def train_step(
        optimizer: flax.optim.Optimizer,
        state: flax.core.frozen_dict.FrozenDict,
        batch: Dict[str, jnp.ndarray],
        prng_key: jnp.ndarray,
        learning_rate_fn: Callable[[int], float],
        l2_reg: float,
        apply_fn : Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Tuple[flax.optim.Optimizer, flax.core.frozen_dict.FrozenDict,
            Dict[str, float], float]:
    """
        Train the model for one step.

        Args:
            optimizer : the optimizer used to train models.
            state : the current state of the model, generally referring to the
              statistics in BN.
            batch : sample batch that used for optimization. 
            prng_key : PRNG Key for the stochasticity in the model.
            learning_rate_fn : learning rate function computing the current
              learning rate for a given step, "f(step) -> lr".
            l2_reg : coefficient of weight l2 regularization. 
            apply_fn : the model.apply function, which applies a module method
              to variables and returns output and modified variables.

        Returns:
            new_optimizer : the updated new optimizer. The optimizer will
              record the updated parameters as the "target" property.
            new_state : the updated new state of the model.
            metrics : the computed metrics that will be recorded in tensorboard.
            lr : the current learning rate.
    """
    # Split the rng for dropout regularization and shake regularization.
    dropout_rng, shake_rng = jax.random.split(prng_key, 2)

    def forward_and_loss(params: flax.core.frozen_dict.FrozenDict,
                         true_gradient = False) \
                         -> Tuple[float,
                                Tuple[flax.core.frozen_dict.FrozenDict, jnp.ndarray]]:

        """
            Define the forward propagation to compute the loss and update state.
              This will also be traced by later backfoward propagation.

            Args:
                params : the current parameters of the model.
                true_gradient : if true, the same mixing parameter will be used
                  for forward and backward propagation in Shake related
                  regularization.
                
            Returns:
                loss : the loss value for the sample batch.
                state : the updated new state of the model based on the sample
                  batch.
                logits : the predicted logit for ths sample batch.
        """                
        
        # If computing the true gradient in shake related regularizations, the
        # true gradient flag will be parsed into apply_fn, else given the
        # defualt value of true gradient is False, we will ignore this arg to
        # use the default value and also to avoid causing argument error for
        # common models.
        if true_gradient:
            logits, new_state = apply_fn(
                variables = {"params":params, **state},
                inputs = batch['image'],
                train = True,
                rngs = dict(dropout = dropout_rng, shake = shake_rng),
                mutable = list(state.keys()),
                true_gradient = true_gradient
            )
        else:
            logits, new_state = apply_fn(
                variables = {"params":params, **state},
                inputs = batch['image'],
                train = True,
                rngs = dict(dropout = dropout_rng, shake = shake_rng),
                mutable = list(state.keys()),
            ) 

        # Calculate the cross entropy loss. If needed, change it to other loss
        # function.
        loss = utli.cross_entropy_loss(logits, batch['label'])
        # L2-Reg will be imposed on all parameters including weight, bias as
        # well as batch norm parameters by default.
        weight_penalty_params = jax.tree_leaves(params)
        if FLAGS.config.no_weight_decay_on_bn:
            weight_l2 = sum(
                [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        else:
            weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])

        # Final loss, the summation of cross entropy and L2-Reg
        weight_penalty = l2_reg * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, (new_state, logits)

    def get_gnp_gradient(params: flax.core.frozen_dict.FrozenDict,
                         *, r: float, alpha : float):
        """
            Compute the gradient where loss is added additional with gradient
              norm penalty,
                    L(theta) = L(theta) + ||\\nabla L(theta)||_2

            See https://arxiv.org/abs/2202.03599 for more details.

            Args:
                params : the current parameters of the model.
                r : a small scalar used for approximating the Hessian
                  multiplication.
                alpha :  alpha = lambda/r; lambda is the coefficient of the
                  gradient norm penalty. In our paper, the gradient of gradient
                  norm penalty will finally be computed by the interplotation
                  between the gradient at the perturbed model and the gradient
                  at the reference model.
                           g = (1 - alpha) * g1 + alpha * g2
                  g1 is the gradient at the reference model (theta). g2 is the
                  gradient at the perturbed model (theta + theta'). See the
                  paper for more detals.

            Returns:
                inner_state : the updated new state of the model based on the
                  sample batch. 
                logits : the predicted logit for ths sample batch.
                g : the gradient of the loss with additional gradient norm
                  penalty.
        """
        # Get the flag that whether true gradient is needed to compute.
        true_gradient_flag = FLAGS.config.has_true_gradient
        # Compute the gradient at the reference model for the sample batches.
        (_, (inner_state, _)), grad = jax.value_and_grad(
            lambda m: forward_and_loss(m, true_gradient = true_gradient_flag), has_aux=True
            )(params)
        # Choose to sync the grad before computing the gradient at the reference
        # model in the second step for gradient computing.
        if FLAGS.config.gnp.sync_perturbations:
            grad = jax.lax.pmean(grad, 'batch')
        grad_origial = grad
        # Compute the dual norm.
        grad = utli.dual_vector(grad)
        # Get the perturbed model theta = theta + r * g1/||g1||_2
        noised_model = jax.tree_multimap(lambda a, b: a + r * b,
                                        params, grad)
        # Get the gradient at the perturbed model.
        (_, (_, logits)), grad_noised = jax.value_and_grad(
                forward_and_loss, has_aux=True
            )(noised_model)
        # If this flag set true, the interplotation will be bewteen the normed
        # gradient at the reference model and the gradient at the perturbed model.
        if FLAGS.config.gnp.norm_perturbations:
            g = jax.tree_multimap(lambda a, b: (1 - alpha) * a + alpha * b, grad, grad_noised)
        else:
            g = jax.tree_multimap(lambda a, b: (1 - alpha) * a + alpha * b, grad_origial, grad_noised)
        return (inner_state, logits), g

    step = optimizer.state.step
    lr = learning_rate_fn(step)
    r = FLAGS.config.gnp.r
    alpha = FLAGS.config.gnp.alpha

    # Use standard training if r eqauls to 0.
    if r != 0: 
        (new_state, logits), grad = get_gnp_gradient(optimizer.target, r = r, alpha = alpha)
    else:
        (_, (new_state, logits)), grad = jax.value_and_grad(
            forward_and_loss, has_aux=True)(
                optimizer.target)

    # Average and Sync the gradient between all devices.
    grad = jax.lax.pmean(grad, 'batch')
    # Clip the gradient after being synchronized.
    grad = utli.clip_by_global_norm(grad)
    # Apply the gradient to optimizer to update parameters.
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    # Compute some metrics to log on tensorboard.
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)]))
    param_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(
            new_optimizer.target)]))
    metrics = {'train_error_rate': utli.error_rate_metric(logits, batch['label']),
                'train_loss': utli.cross_entropy_loss(logits, batch['label']),
                'gradient_norm': gradient_norm,
                'param_norm': param_norm}

    return new_optimizer, new_state, metrics, lr


def eval_step(params: flax.core.frozen_dict.FrozenDict,
              state: flax.core.frozen_dict.FrozenDict,
              batch: Dict[str, jnp.ndarray],
              apply_fn : Callable[[jnp.ndarray], jnp.ndarray])\
               -> Dict[str, float]:
    """
        Evaluate the model on the given sample batch.

        Args:
            params : the current parameters of the model.
            state : the current state of the model, generally referring to the
              statistics in BN.
            batch : sample batch that used for evaluating.
            apply_fn : the model.apply function, which applies a module method
              to variables and returns output and modified variables.

        Returns:
            metrics : the computed evaluation metrics that will be recorded in
              tensorboard.
    """

    # Average and Sync the state between all devices. 
    state = jax.lax.pmean(state, 'batch')
    # Compute the predicted logits based on the parameters and state. During
    # evaluation, all the parameters and states will be inmuatble.
    logits = apply_fn(
        variables = {"params" : params, **state},
        train = False,
        mutable = False,
        inputs = batch['image']
    )
    # Since the dataset pipeline mimics that in SAM repo, we will follow the
    # same mask tech to tackle samples that is unaligned to the batch size.
    # From SAM repo:
    #   Because we don't have a guarantee that all batches contains the same number
    #    of samples, we can't average the metrics per batch and then average the
    #    resulting values. To compute the metrics correctly, we sum them (error rate
    #    and cross entropy returns means, thus we multiply by the number of samples),
    #    and finally sum across replicas. These sums will be divided by the total
    #    number of samples outside of this function.
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
    # Sync by summing the metrics over all the devices.
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def eval_on_dataset(param: flax.core.frozen_dict.FrozenDict,
                    state: flax.core.frozen_dict.FrozenDict,
                    dataset : dataset_source.DatasetSource,
                    pmapped_eval_step : Callable[
                        [flax.core.frozen_dict.FrozenDict,
                         flax.core.frozen_dict.FrozenDict,
                         Dict[str, jnp.ndarray],
                         Callable[[jnp.ndarray], jnp.ndarray]],
                         Dict[str, float]
                    ],
    ) -> Dict[str, float]:
    """
        Evaluate the model on the given testing dataset.

        Args:
            params : the current parameters of the model.
            state : the current state of the model, generally referring to the
              statistics in BN.
            dataset : the dataset pipeline.
            pmapped_eval_step : the pmapped eval_step function, which is the
              function after mapping by jax.pmap() for parallel execution.

        Returns:
            eval_summary : the computed evaluation metrics that will be recorded
              in tensorboard.
    """    

    eval_metrics = []
    total_num_samples = 0
    all_host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')
    for eval_batch in dataset:
        # Convert the sample from tensorflow Dataset object to numpy.
        eval_batch = utli.tensorflow_to_numpy(eval_batch)
        # Shard the sample batches to fit the number of gpus.
        eval_batch = utli.shard_batch(eval_batch)
        # Evaluate for one batch.
        metrics = pmapped_eval_step(param, state, eval_batch)
        # Collect the evaluation metrics.
        eval_metrics.append(metrics)
        # Tackle the samples that are unaligned to the number of devices.
        if 'mask' not in eval_batch:
            total_num_samples += (
                eval_batch['label'].shape[0] * eval_batch['label'].shape[1] *
                jax.host_count())
        else:
            total_num_samples += all_host_psum(eval_batch['mask'])[0].sum()

    # Since the metric is sync and summed up over all the devices, the metrics
    # in each device is exactly the same, i.e. the metrics summed over all
    # samples. So firstly, reduce the values in each replicas to a single copy.
    # Then, divide the metrics by the number of samples to compute the average
    # metrics over the testing set.
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_summary = jax.tree_map(lambda x: x.sum() / total_num_samples,
                                eval_metrics)
    return eval_summary


_Mapped_Train_Func = Callable[
    [
        flax.optim.Optimizer,
        flax.core.frozen_dict.FrozenDict,
        Dict[str, jnp.ndarray],
        jnp.ndarray,
        Callable[[int], float],
        float,
        Callable[[jnp.ndarray], jnp.ndarray]
    ],
        Tuple[flax.optim.Optimizer, flax.core.frozen_dict.FrozenDict,
                Dict[str, float], float]
]

def train_for_one_epoch(
        optimizer: flax.optim.Optimizer,
        state: flax.core.frozen_dict.FrozenDict,
        dataset_source : dataset_source.DatasetSource,
        prng_key: jnp.ndarray,
        pmapped_train_step : _Mapped_Train_Func,
    ) -> Tuple[flax.optim.Optimizer, flax.core.frozen_dict.FrozenDict, Dict[str, float]]:

    """
        Train the model on the given training dataset.

        Args:
            params : the current parameters of the model.
            state : the current state of the model, generally referring to the
              statistics in BN.
            dataset : the dataset pipeline.
            prng_key : PRNG Key for the stochasticity in the model.
            pmapped_train_step : the pmapped pmapped_train_step function, which
              is the function after mapping by jax.pmap() for parallel execution.

        Returns:
            optimizer : the optimizer used to train models.
            state : the current state of the model, generally referring to the
              statistics in BN.
            train_summary : the computed evaluation metrics that will be recorded
              in tensorboard.
    """  

    train_metrics = []
    # Start training on each sample batch.
    for batch in dataset_source.get_train(use_augmentations=True):
        # Generate a particular PRNG by combing the current training step.
        step_key = jax.random.fold_in(prng_key, optimizer.state.step[0])
        # Convert the sample from tensorflow Dataset object to numpy.
        batch = utli.tensorflow_to_numpy(batch)
        # Shard the sample batches to fit the number of gpus.
        batch = utli.shard_batch(batch)
        # Shard the Prng such that each device would receive a unique key.
        sharded_keys = common_utils.shard_prng_key(step_key)
        # Training the sample batch.
        optimizer, state, metrics, lr = pmapped_train_step(
            optimizer, state, batch, sharded_keys)
        # Collect the training metric summaries from all devices.
        train_metrics.append(metrics)
    # Reduce the metrics in all devices to a single copy. 
    train_metrics = common_utils.get_metrics(train_metrics)
    # Average the training summaries within this epoch.
    train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
    train_summary['learning_rate'] = lr[0]

    return optimizer, state, train_summary
