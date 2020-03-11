"""
    Credits: https://github.com/sfujim/TD3
"""

import argparse
from typing import Any, Tuple
import numpy as np

import gym
import jax

from replay_buffer import ReplayBuffer
from agent import Agent

import os

OptState = Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", choices=['TD3', 'DDPG'])  # Policy name (TD3, DDPG)
    parser.add_argument("--env", default="InvertedPendulum-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", type=int, required=True)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--replay_size", default=200000, type=int)  # Size of the replay buffer
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--lr", default=3e-4, type=float)  # Optimizer learning rates
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    # TODO: Model saving and loading is not supported yet.
    # parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    # parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    return args


def eval_policy(agent: Agent, env_name: str, eval_episodes: int = 10, max_steps: int = 0) -> float:
    eval_env = gym.make(env_name)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        remaining_steps = max_steps * 1.0

        while not done:
            action = agent.policy(agent.actor_params, state)
            state, reward, done, _ = eval_env.step(action)

            remaining_steps -= 1

            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main():
    args = parse_arguments()

    idx = 0
    file_name = f"{args.env}_{idx}"
    # For easy extraction of the data, we save all runs using a serially increasing indicator.
    while os.path.exists('./results/' + args.policy + '/' + file_name + '.npy'):
        idx += 1
        file_name = f"{args.env}_{idx}"

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results/" + args.policy):
        os.makedirs("./results/" + args.policy)

    # if args.save_model and not os.path.exists("./models/" + args.policy):
    #     os.makedirs("./models/" + args.policy)

    env = gym.make(args.env)
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    rng = jax.random.PRNGKey(args.seed)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    agent = Agent(args.policy,
                  action_dim,
                  max_action,
                  args.lr,
                  args.discount,
                  args.noise_clip,
                  args.policy_noise,
                  args.policy_freq,
                  actor_rng,
                  critic_rng,
                  state)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=args.replay_size)

    # Evaluate untrained policy.
    # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
    evaluations = [eval_policy(agent, args.env, max_steps=env._max_episode_steps, eval_episodes=100)]
    np.save(f"./results/{args.policy}/{file_name}", evaluations)
    best_performance = evaluations[-1]
    best_actor_params = agent.actor_params
    # if args.save_model: agent.save(f"./models/{args.policy}/{file_name}")

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            rng, noise_rng = jax.random.split(rng)
            action = (
                    agent.policy(agent.actor_params, state)
                    + jax.random.normal(noise_rng, (action_dim, )) * max_action * args.expl_noise
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem we are
        # solving, however it has been observed empirically to work pretty well.
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            rng, update_rng = jax.random.split(rng)
            agent.update(replay_buffer, args.batch_size, update_rng)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(agent, args.env, max_steps=env._max_episode_steps, eval_episodes=100))
            np.save(f"./results/{args.policy}/{file_name}", evaluations)
            if evaluations[-1] > best_performance:
                best_performance = evaluations[-1]
                best_actor_params = agent.actor_params
                # if args.save_model: agent.save(f"./models/{args.policy}/{file_name}")

    # At the end, re-evaluate the policy which is presumed to be best. This ensures an un-biased estimator when
    # reporting the average best results across each run.
    agent.actor_params = best_actor_params
    evaluations.append(eval_policy(agent, args.env, max_steps=env._max_episode_steps, eval_episodes=100))
    np.save(f"./results/{args.policy}/{file_name}", evaluations)
    print(f"Selected policy has an average score of: {evaluations[-1]:.3f}")


if __name__ == "__main__":
    main()
