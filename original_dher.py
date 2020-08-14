import ipdb
import argparse
import datetime
import itertools

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gym_line_follower
import wandb
from gym_line_follower.envs import LineFollowerEnv
from replay_memory import ReplayGMemory, ReplayMemory
from sac import SAC
from utils import get_goal, get_her_goal


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default='FetchReach-v1',
                    help='Mujoco Gym environment (default: FetchReach-v1)')
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

wandb.init(name=f"{args.env_name}-DHER", project="MyExp")
# Environment
env = gym.make(args.env_name)

env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Agent
if args.env_name.startswith('Fetch'):
    env_space = env.observation_space.spaces
    agent = SAC(
        env_space['observation'].shape[0]+env_space['desired_goal'].shape[0],
        env.action_space, args)
else:
    agent = SAC(
        env.observation_space.shape[0]+2,
        env.action_space, args)

# Memory
memory = ReplayGMemory(args.replay_size, args.seed)
achieved_hash = dict()
desired_hash = dict()
failed_episodes = list()

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
    goal = state['desired_goal']
    her_goal = state['achieved_goal']
    state = state['observation']
    if did_it:
        did_it = False
    while not done:
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
        next_her_goal = next_state['achieved_goal']
        next_state = next_state['observation']

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = float(not done)
        # Append transition to memory
        memory.push(state, action, reward, next_state, mask, goal)
        episode.append((state, action, reward, mask,
                        next_state, goal, her_goal, next_her_goal))

        state = next_state
        her_goal = next_her_goal
    
    if reward < -10:
        failed_episodes.append(episode)

    if len(failed_episodes) > 1:
        # Ej eh o episodio que ta chegando
        Ej = failed_episodes[-1]
        last_experience = Ej[-1]
        Ej_desired_goal = last_experience[-3]
        for i, Ei in enumerate(failed_episodes[:-1]):
            # Linha 15 do algoritmo
            last_experience = Ei[-1]
            Ei_achieved_goal = last_experience[-1]
            is_good = Ei_achieved_goal.all() == Ej_desired_goal.all()
            # Linha 15 do algoritmo

            # Linha 16 do algoritmo
            if is_good:
                p = len(Ei)
                q = len(Ej)
                m = min(p, q)
                # Linha 17 do algoritmo
                Ej_desired_goals = [exp[-2] for exp in Ej[:m]]
                # Linha 18
                for t in range(m-1):
                    bef_new_goal = Ej_desired_goals[t]
                    new_goal = Ej_desired_goals[t+1]
                    # O rework do reward eh no episodio Ei que tem tamanho p=len(Ei) e so pode ir ate o tamanho  m=min(len(Ei),len(Ej))
                    state, action, _, next_state, done, _, _, next_achieved_goal = Ei[p - m + t]
                    # O estado S[p-m+t] eh concatenado com o g't
                    state = np.concatenate((state, bef_new_goal))
                    # O estado S[p-m+t+1] eh concatenado com o g'(t+1)
                    next_state = np.concatenate((next_state, new_goal))

                    # Linha 19
                    # Reward eh calculado com o g'(t+1)
                    # O calculo do environment em questao leva em conta o next_achieved_goal, que no caso eh a proxima posicao robo
                    new_reward = env.compute_reward(
                        next_achieved_goal, new_goal, None)
                    memory.push(state, action, new_reward,
                                next_state, done, new_goal)

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
            goal = state['desired_goal']
            her_goal = state['achieved_goal']
            state = state['observation']
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(np.concatenate(
                    [state, goal]), evaluate=True)
                next_state, reward, done, robot_pos = env.step(action)
                next_her_goal = next_state['achieved_goal']
                next_state = next_state['observation']
                episode_reward += reward

                state = next_state
                her_goal = next_her_goal
            avg_reward += episode_reward
        avg_reward /= episodes

        wandb.log({'reward_test': avg_reward})

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        agent.save_model(env_name=args.env_name, suffix='her')

env.close()
