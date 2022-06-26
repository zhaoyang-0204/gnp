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

"""Adam optimizer."""

import flax
import jax
import jax.numpy as jnp
import numpy as np


class AdamOptimizer(flax.optim.OptimizerDef):
    """Adam optimizer with gradient clip."""

    @flax.struct.dataclass
    class HyperParams:
        learning_rate: np.ndarray
        beta1:np.ndarray
        beta2:np.ndarray
        eps:np.ndarray
        weight_decay:np.ndarray
        grad_norm_clip:np.ndarray

    @flax.struct.dataclass
    class State:
        grad_ema:np.ndarray
        grad_sq_ema:np.ndarray

    def __init__(self,
                learning_rate = None,
                beta1 = 0.9,
                beta2 = 0.999,
                eps = 1e-8,
                weight_decay = 0.0,
                grad_norm_clip = 1.0):

        hyper_params = AdamOptimizer.HyperParams(learning_rate = learning_rate,
                                                beta1 = beta1,
                                                beta2 = beta2,
                                                eps = eps,
                                                weight_decay = weight_decay,
                                                grad_norm_clip = grad_norm_clip)
        super().__init__(hyper_params)
    
    def init_param_state(self, param):
        return AdamOptimizer.State(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_gradient(self, hyper_params, params, state, grads):
        step = state.step
        params_flat, treedef = jax.tree_flatten(params)
        states_flat = treedef.flatten_up_to(state.param_states)
        grads_flat = treedef.flatten_up_to(grads)

        if hyper_params.grad_norm_clip:
            grads_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
            grads_factor = jnp.minimum(1.0, hyper_params.grad_norm_clip / grads_l2)
            grads_flat = jax.tree_map(lambda param: grads_factor * param, grads_flat)

        out = [
            self.apply_param_gradient(step, hyper_params, param, state, grad)
            for param, state, grad in zip(params_flat, states_flat, grads_flat)
        ]

        new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
        new_params = jax.tree_unflatten(treedef, new_params_flat)
        new_param_states = jax.tree_unflatten(treedef, new_states_flat)
        new_state = flax.optim.OptimizerState(step + 1, new_param_states)
        return new_params, new_state


    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        grad_sq = jax.lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = jnp.array(step + 1, jax.lax.dtype(param.dtype))
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
        new_param -= hyper_params.learning_rate * weight_decay * param
        new_state = AdamOptimizer.State(grad_ema, grad_sq_ema)
        return new_param, new_state
