from typing import Any, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix
import rlax
import numpy as np
from networks import Actor, Critic
import functools
import replay_buffer as rb

OptState = Any


@jax.jit
def soft_update(target_params: hk.Params, online_params: hk.Params, tau: float = 0.005) -> hk.Params:
    return jax.tree_multimap(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)


class Agent(object):
    def __init__(
            self,
            action_dim,
            max_action,
            lr,
            discount,
            noise_clip,
            policy_noise,
            policy_freq,
            actor_rng,
            critic_rng,
            sample_state
    ):
        self.discount = discount
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq

        self.actor = hk.transform(lambda x: Actor(action_dim, max_action)(x))
        actor_opt_init, self.actor_opt_update = optix.adam(lr)

        self.critic = hk.transform(lambda x: Critic()(x))
        critic_opt_init, self.critic_opt_update = optix.adam(lr)

        self.actor_params = self.target_actor_params = self.actor.init(actor_rng, sample_state)
        self.actor_opt_state = actor_opt_init(self.actor_params)

        action = self.actor.apply(self.actor_params, sample_state)

        self.critic_params = self.target_critic_params = self.critic.init(critic_rng, jnp.concatenate((sample_state, action), 0))
        self.critic_opt_state = critic_opt_init(self.critic_params)

        self.updates = 0

    def update(
            self,
            replay_buffer: rb.ReplayBuffer,
            batch_size: int,
            rng
    ) -> Tuple[hk.Params, hk.Params, OptState, hk.Params, hk.Params, OptState]:
        self.updates += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        self.critic_params, self.critic_opt_state = self.update_critic(self.critic_params, self.target_critic_params,
                                                                       self.target_actor_params, self.critic_opt_state,
                                                                       state, action, next_state, reward, not_done, rng)

        if self.updates % self.policy_freq == 0:
            self.actor_params, self.actor_opt_state = self.update_actor(self.actor_params, self.critic_params,
                                                                        self.actor_opt_state, state)

            self.target_actor_params = soft_update(self.target_actor_params, self.actor_params)
            self.target_critic_params = soft_update(self.target_critic_params, self.critic_params)

    @functools.partial(jax.jit, static_argnums=0)
    def critic_1(
            self,
            critic_params: hk.Params,
            state_action: np.ndarray
    ) -> jnp.DeviceArray:
        return self.critic.apply(critic_params, state_action)[0].squeeze(-1)

    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            state: np.ndarray
    ) -> jnp.DeviceArray:
        action = self.actor.apply(actor_params, state)
        return - jnp.mean(self.critic_1(critic_params, jnp.concatenate((state, action), 1)))

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            actor_opt_state: OptState,
            state: np.ndarray
    ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        _, gradient = jax.value_and_grad(self.actor_loss)(actor_params, critic_params, state)
        updates, opt_state = self.actor_opt_update(gradient, actor_opt_state)
        new_params = optix.apply_updates(actor_params, updates)
        return new_params, opt_state

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
            self,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            target_actor_params: hk.Params,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: np.ndarray,
            not_done: np.ndarray,
            rng
    ) -> jnp.DeviceArray:
        noise = (
                jax.random.normal(rng, shape=action.shape) * self.policy_noise
        ).clip(-self.noise_clip, self.noise_clip)

        next_action = (
                self.actor.apply(target_actor_params, next_state) + noise
        ).clip(-self.noise_clip, self.noise_clip)

        next_q_1, next_q_2 = self.critic.apply(target_critic_params, jnp.concatenate((next_state, next_action), 1))
        target_q = jax.lax.stop_gradient(reward + self.discount * jax.lax.min(next_q_1, next_q_2) * not_done)
        q_1, q_2 = self.critic.apply(critic_params, jnp.concatenate((state, action), 1))

        return jnp.mean(rlax.l2_loss(q_1, target_q) + rlax.l2_loss(q_2, target_q))

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(
            self,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            target_actor_params: hk.Params,
            critic_opt_state: OptState,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: np.ndarray,
            not_done: np.ndarray,
            rng
    ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        _, gradient = jax.value_and_grad(self.critic_loss)(critic_params, target_critic_params, target_actor_params,
                                                           state, action, next_state, reward, not_done, rng)
        updates, opt_state = self.critic_opt_update(gradient, critic_opt_state)
        new_params = optix.apply_updates(critic_params, updates)
        return new_params, opt_state

    @functools.partial(jax.jit, static_argnums=0)
    def policy(self, actor_params: hk.Params, state: np.ndarray) -> jnp.DeviceArray:
        return self.actor.apply(actor_params, state)
