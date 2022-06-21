import jax.numpy as jnp
import jax
import math
from flax.training import lr_schedule



def create_exponential_learning_rate_schedule(
    base_learning_rate: float,
    steps_per_epoch: int,
    lamba: float,
    warmup_epochs: int = 0):
    def learning_rate_fn(step):
        t = step / steps_per_epoch
        return base_learning_rate * jnp.exp(-t / lamba) * jnp.minimum(
            t / warmup_epochs, 1)

    return learning_rate_fn


def get_cosine_schedule(num_epochs: int, learning_rate: float,
                        num_training_obs: int,
                        batch_size: int):

    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
        learning_rate, steps_per_epoch // jax.host_count(), num_epochs,
        warmup_length=0)

    return learning_rate_fn

def get_exponential_schedule(num_epochs: int, learning_rate: float,
                             num_training_obs: int,
                             batch_size: int):

    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    # At the end of the training, lr should be 1.2% of original value
    # This mimic the behavior from the efficientnet paper.
    end_lr_ratio = 0.012
    lamba = - num_epochs / math.log(end_lr_ratio)
    learning_rate_fn = create_exponential_learning_rate_schedule(
        learning_rate, steps_per_epoch // jax.host_count(), lamba)
    return learning_rate_fn