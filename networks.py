from typing import Any, Tuple
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

"""
    Actor and Critic networks defined as in the TD3 paper (Fujimoto et. al.) https://arxiv.org/abs/1802.09477
"""


class Actor(hk.Module):
    def __init__(self, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action

    def __call__(self, state: np.ndarray) -> jnp.DeviceArray:
        actor_net = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(self.action_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform'))
        ])
        return jnp.tanh(actor_net(state)) * self.max_action


class Critic(hk.Module):
    def __init__(self):
        super(Critic, self).__init__()

    def __call__(self, state_action: np.ndarray) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        def critic_net():
            return hk.Sequential([
            hk.Flatten(),
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(1, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform'))
        ])
        critic_net_1 = critic_net()

        critic_net_2 = critic_net()

        return critic_net_1(state_action), critic_net_2(state_action)
