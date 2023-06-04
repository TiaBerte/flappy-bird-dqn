"""
Script for visualizing the trained model playing.

"""


import time
import flappy_bird_gym
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer
from argparse import ArgumentParser
from agent import Agent
from typing import List
import numpy as np
import random 
import torch

parser = ArgumentParser()


parser.add_argument("--hidden_dim", help="Network hidden dim list", default=[256, 256], type=List[int])
parser.add_argument("--dueling", help="Flag for using dueling network", action="store_false")
parser.add_argument("--loading_path", help="Path from which to load the trained model", type=str, default='DuelingDQN\DuelingDQN.pt')


args = parser.parse_args()



env = flappy_bird_gym.make("FlappyBird-v0")
renderer = FlappyBirdRenderer(audio_on=False)

renderer.audio_on = False

seed = 0
env.seed(seed)
env.action_space.seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


env._renderer = renderer
env._renderer.make_display()

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Define agent
agent = Agent(state_dim, n_actions, args.dueling)

agent.load_model(args.loading_path)
obs = env.reset()
tot_rew = 0
while True:
    action = agent.act(obs,True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1 / 60) 
    tot_rew += reward 
    if done:
        break
    
print('reward', tot_rew)