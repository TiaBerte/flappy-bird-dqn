import numpy as np
import torch
import random
from agent import Agent
import flappy_bird_gym 
from tqdm import tqdm
import os 
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from typing import List
import yaml

parser = ArgumentParser()

parser.add_argument("--training", help="Flag training model", action="store_false")
parser.add_argument("--testing", help="Flag for testing the trained model", action="store_true")
parser.add_argument("--max_episodes", help="Max number of episodes", default=2500, type=int)
parser.add_argument("--eval_episodes", help="Number of episodes to average during the evaluation", default=10, type=int)
parser.add_argument("--max_episode_steps", help="Max number of step per episodes", default=1000, type=int)
parser.add_argument("--eval_freq", help="Number of training episodes after which the model is tested", default=10, type=int)


# Replay buffer args
parser.add_argument("--buffer_capacity", help="Replay buffer capacity", default=100000, type=int)
parser.add_argument("--batch_size", help="batch size", default=256, type=int)
parser.add_argument("--per", help="Flag for using prioritized experience replay", action="store_true")
parser.add_argument("--alpha", help="Alpha hyperparameter of PER", default=0.6, type=float)
parser.add_argument("--beta_start", help="Initial value of beta hyperparameter of PER", default=0.4, type=float)
parser.add_argument("--end_beta_incr", help="Episode in which to finish the increasing of beta", default=2250, type=int)


# DQN parameters
parser.add_argument("--dueling", help="Flag for using dueling network", action="store_true")
parser.add_argument("--gamma", help="Discount reward value", default=0.99, type=float)
parser.add_argument("--tau", help="Soft update parameter", default=0.005, type=float)
parser.add_argument("--lr", help="Network learning rate", default=5e-5, type=float)
parser.add_argument("--hidden_dim", help="Network hidden dim list", default=[256, 256], type=List[int])
parser.add_argument("--max_eps", help="Starting value of epsilon parameter", default=0.1, type=float)
parser.add_argument("--min_eps", help="Final value of epsilon parameter", default=0.01, type=float)
parser.add_argument("--decay_ep", help="Number of episode during which eps decrease linearly from max eps to min eps", default=2250, type=float)


args = parser.parse_args()



def main(args):
    
    env = flappy_bird_gym.make("FlappyBird-v0")

    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(state_dim, n_actions, args.dueling, args.hidden_dim, args.buffer_capacity, args.batch_size, args.gamma,
                         args.tau, args.lr, args.max_eps, args.min_eps, args.decay_ep, args.per, args.alpha, args.beta_start, args.end_beta_incr)

    print(f'Selected {agent.version} model.')
    
    if not os.path.exists(f'./{agent.version}'):
        os.makedirs(f'./{agent.version}')

    if args.training:

        with open(f'./{agent.version}/{agent.version}.yaml', 'w') as file:
            yaml.dump(args, file)

        writer_path = os.path.join(f"{agent.version}")
        writer = SummaryWriter(writer_path)

        state = env.reset()
        for i in range(args.buffer_capacity):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            agent.buffer.add(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state

        train_steps = 0
        best_test_reward = 0
        avg_train_reward = 0

        for episode in tqdm(range(1, args.max_episodes + 1)):
            state = env.reset()
            episode_reward = 0
            done = False
            for _ in range(args.max_episode_steps):
                action = agent.act(state, eval_mode=False)
                next_state, reward, done, info = env.step(action)
                agent.buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                loss = agent.update_model()
                writer.add_scalar("Loss", loss, train_steps)
                train_steps += 1
                agent.update_target()

                if done:
                    break
            
            agent.update_epsilon(episode)
            if agent.per:
                agent.buffer.update_beta(episode)
                writer.add_scalar("Beta PER", agent.buffer.beta, episode)
                
            avg_train_reward = avg_train_reward + (episode_reward - avg_train_reward)/episode
            writer.add_scalar("Epsilon", agent.eps, episode)
            writer.add_scalar("Train reward", episode_reward, episode)
            writer.add_scalar("Average train reward", avg_train_reward, episode)

            if episode % args.eval_freq == 0:
                test_rewards = []
                for eval_episode in range(args.eval_episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent.act(state, eval_mode=True)
                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        state = next_state

                    test_rewards.append(episode_reward)
                avg_test_reward = sum(test_rewards) / args.eval_episodes
                writer.add_scalar("Average test reward", avg_test_reward, episode)

                if avg_test_reward > best_test_reward:
                    print('\nSaving model ...')
                    model_path = os.path.join(f"{agent.version}/{agent.version}.pt")
                    agent.save_model(model_path)
                    print(f'Saved model at {model_path} !')
                    best_test_reward = avg_test_reward

    if args.testing:
        model_path = os.path.join(f"{agent.version}/{agent.version}.pt")
        print(f'Loading model weights from {model_path} ...')
        agent.load_model(model_path)
        print(f'Model weights loaded !')
        test_rewards = []
        for eval_episode in tqdm(range(args.eval_episodes)):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.act(state, eval_mode=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

            test_rewards.append(episode_reward)
        avg_test_reward = sum(test_rewards) / args.eval_episodes
        print('Average test reward ', avg_test_reward)

main(args)
