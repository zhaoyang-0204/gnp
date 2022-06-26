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
    Get the learning rate schedule.
"""

import jax.numpy as jnp
import jax
import math
from flax.training import lr_schedule
from typing import Optional, List, Callable

def get_learning_schedule(lr_schedule_type : str,
                          base_lr : float,
                          num_epochs : int,
                          num_trainig_samples : int,
                          batch_size : int,
                          warmup_epochs : Optional[int] = 0,
                          stepped_schedule : Optional[List] = None,
                          ) -> Callable[[int], float]:
    """
        Get the learning rate schedule. Available schedules are 

        Args:
            lr_schedule_type : the type of learning rate schdule.
            base_lr : the base learning rate.
            num_epochs : the total number of training epochs.
            num_trainig_samples : the number of training samples.
            batch_size : the batch size.
            warmup_epochs : the epochs to perform warmup operation.
            stepped_schedule : this would work only if the learning rate is
              scheduled as stepped. The arg is a list where each element
              sequentially indicates a specific learning rate at a specific
              epoch. It should follow the following format :
                [
                [30, 0.1],
                [60, 0.01],
                [80, 0.001]
                ]

        Returns:
            learning_rate_fn : learning rate function computing the current
              learning rate for a given step, "f(step) -> lr".
    """
    _AVAILABLE_LR_SCHEDULE = (
        "constant", "stepped", "cosine", "exponential",
    )
    if lr_schedule_type not in _AVAILABLE_LR_SCHEDULE:
        lr_schedule_type = "constant"

    steps_per_epoch = int(math.floor(num_trainig_samples / batch_size))
    if lr_schedule_type == "constant":
        learning_rate_fn = lr_schedule.create_constant_learning_rate_schedule(
            base_learning_rate=base_lr,
            steps_per_epoch=steps_per_epoch,
            warmup_length=warmup_epochs
        )
    elif lr_schedule_type == "stepped":
        assert stepped_schedule is not None
        learning_rate_fn = lr_schedule.create_stepped_learning_rate_schedule(
            base_learning_rate=base_lr,
            steps_per_epoch=steps_per_epoch,
            lr_sched_steps=stepped_schedule,
            warmup_length=warmup_epochs
        )
    elif lr_schedule_type == "cosine":
        learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
            base_learning_rate = base_lr,
            steps_per_epoch = steps_per_epoch,
            halfcos_epochs = num_epochs,
            warmup_length=warmup_epochs)
    elif lr_schedule_type == "exponential":
        learning_rate_fn = get_exponential_schedule(
            num_epochs = num_epochs,
            learning_rate=base_lr,
            num_training_obs=num_trainig_samples,
            batch_size=batch_size
            )
    else:
        raise ValueError("Unkown learning schedule type!")
 
    return learning_rate_fn


"""
    Exponential learning rate schedule is forked from the SAM repo
    https://github.com/google-research/sam.
"""

def create_exponential_learning_rate_schedule(
    base_learning_rate: float,
    steps_per_epoch: int,
    lamba: float,
    warmup_epochs: int = 0):
    """Creates a exponential learning rate schedule with optional warmup.

    Args:
        base_learning_rate: The base learning rate.
        steps_per_epoch: The number of iterations per epoch.
        lamba: Decay is v0 * exp(-t / lambda).
        warmup_epochs: Number of warmup epoch. The learning rate will be modulated
        by a linear function going from 0 initially to 1 after warmup_epochs
        epochs.

    Returns:
        Function `f(step) -> lr` that computes the learning rate for a given step.
    """
    def learning_rate_fn(step):
        t = step / steps_per_epoch
        return base_learning_rate * jnp.exp(-t / lamba) * jnp.minimum(
            t / warmup_epochs, 1)

    return learning_rate_fn


def get_exponential_schedule(num_epochs: int, learning_rate: float,
                             num_training_obs: int,
                             batch_size: int):
    """Returns an exponential learning rate schedule, without warm up.

    Args:
        num_epochs: Number of epochs the model will be trained for.
        learning_rate: Initial learning rate.
        num_training_obs: Number of training observations.
        batch_size: Total batch size (number of samples seen per gradient step).

    Returns:
        A function that takes as input the current step and returns the learning
        rate to use.
    """
    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    # At the end of the training, lr should be 1.2% of original value
    # This mimic the behavior from the efficientnet paper.
    end_lr_ratio = 0.012
    lamba = - num_epochs / math.log(end_lr_ratio)
    learning_rate_fn = create_exponential_learning_rate_schedule(
        learning_rate, steps_per_epoch // jax.host_count(), lamba)
    return learning_rate_fn