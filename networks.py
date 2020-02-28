from typing import Any, Tuple
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np


class Actor(hk.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action

    def __call__(self, state):
        actor_net = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(256, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.relu,
            hk.Linear(self.action_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
            jax.nn.sigmoid
        ])
        return (actor_net(state) * 2 - 1) * self.max_action


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