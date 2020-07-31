import argparse
import datetime
import itertools

import gym
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import gym_line_follower
from gym_line_follower.envs import LineFollowerEnv
from replay_memory import ReplayGMemory, ReplayMemory
from sac import SAC

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
env_test = LineFollowerEnv(gui=True, sub_steps=10, max_track_err=0.05,
                           max_time=60, power_limit=0.99)

env_test.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
# Normal
# agent = SAC(env_test.observation_space.shape[0], env_test.action_space, args)
# With objective
agent = SAC(env_test.observation_space.shape[0]+4, env_test.action_space, args)
path = 'models/sac_CHANGE_LineFollower-v0_goncaexp'
agent.load_model(path.replace('CHANGE', 'actor'),
                 path.replace('CHANGE', 'critic'))

episodes = 100
avg_reward = 0.
for _ in range(episodes):
    state = env.reset(do_rand=False)
    objectives = np.array(list(zip(env.track.x, env.track.y))[1:])
    robot_pos = env._get_info()
    robot_pos = np.array(list(robot_pos.values()))[:-1]
    state = np.concatenate([state, robot_pos])
    episode_reward = 0
    done = False
    while not done:
        # env.render(mode='human')
        percentage = env.position_on_track/env.track.length
        percentage = percentage if percentage > 0 else 0
        objective_idx = int(objectives.shape[0]*percentage)
        objective = objectives[objective_idx]
        action = agent.select_action(np.concatenate([state, objective]),
                                        evaluate=True)

        next_state, reward, done, robot_pos = env.step(action)  # Step
        robot_pos = np.array(list(robot_pos.values()))[:-1]
        next_state = np.concatenate([next_state, robot_pos])
        episode_reward += reward

        state = next_state
    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(
    episodes, round(avg_reward, 2)))
print("----------------------------------------")


env.close()
