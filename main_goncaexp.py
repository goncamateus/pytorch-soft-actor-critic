import argparse
import datetime
import itertools

import gym
import numpy as np
import torch

import gym_line_follower
import wandb
from gym_line_follower.envs import LineFollowerEnv
from replay_memory import ReplayGMemory, ReplayMemory
from sac import SAC
from utils import get_goal, get_her_goal

wandb.init(name="LineFollower-GoncaExp", project="Cadeira-RL")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="LineFollower-v0",
                    help='Mujoco Gym environment (default: LineFollower-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)

env.seed(args.seed)
# env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0]+3, env.action_space, args)

# Memory
memory = ReplayGMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
did_it = False
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    episode = []
    state = env.reset()
    worst_goals = np.array([[0, 0, 0], [env.track.checkpoints[-1],
                                        360, env.max_track_err]])
    worst_dist = np.linalg.norm(worst_goals[0] - worst_goals[1])
    if did_it:
        did_it = False
    while not done:
        goal = get_goal(env)
        her_goal = get_her_goal(env)
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(np.concatenate([state, goal]))

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory, args.batch_size, updates)

                wandb.log({"critic_1": critic_1_loss,
                           "critic_2": critic_2_loss,
                           "policy": policy_loss,
                           "entropy_loss": ent_loss,
                           "temp_alpha": alpha})
                updates += 1

        next_state, reward, done, robot_pos = env.step(action)  # Step
        next_her_goal = get_her_goal(env)
        if not done:
            reward = 0
        episode_reward += reward
        if env.track.done:
            did_it = True
        episode_steps += 1
        total_numsteps += 1

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = float(not done)
        # Append transition to memory
        # memory.push(state, action, reward, next_state, mask, goal)
        episode.append((state, action, reward, mask,
                        next_state, goal, her_goal, next_her_goal))

        state = next_state
    for i, (state, action, reward, done,
            next_state, goal, her_goal, next_her_goal) in enumerate(episode):
        d1 = np.linalg.norm(her_goal-goal)
        d2 = np.linalg.norm(next_her_goal-goal)
        state_bef = episode[i-1][0]
        action_bef = episode[i-1][1]
        reward_bef = episode[i-1][2]
        done_bef = episode[i-1][3]
        new_reward = 0
        if d2 > d1 and i > 0:
            new_reward = np.exp((-d1/worst_dist))
            memory.push(
                state_bef, action_bef, new_reward, state, done_bef, her_goal)
        elif d1 > d2 and i > 0:
            new_reward = -np.exp((-d1/worst_dist))
            memory.push(
                state_bef, action_bef, new_reward, state, done_bef, goal)
        episode_reward += new_reward
    if total_numsteps > args.num_steps:
        break

    wandb.log({'reward_train': episode_reward})
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
        i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 100 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 3
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(np.concatenate(
                    [state, get_goal(env)]), evaluate=True)
                next_state, reward, done, robot_pos = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        wandb.log({'reward_test': avg_reward})

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        agent.save_model(env_name=args.env_name, suffix='goncaexp')

env.close()
