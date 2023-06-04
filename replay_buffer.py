import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    

class PrioritizedReplay(ReplayBuffer):
    def __init__(self,  capacity, alpha, beta_start, end_beta_incr):
        super().__init__(capacity)
        self.capacity = capacity
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.n_episodes = 0
        self.beta_start = beta_start
        self.beta = beta_start
        self.end_beta_incr = end_beta_incr


    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(float(max(self.priorities)))
        else:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(1)


    def compute_indexes(self, batch_size):
        n_elements = len(self.priorities)
        p_i = np.array(self.priorities) + 1e-8
        probabilities = np.power(p_i, self.alpha)/np.sum(np.power(p_i, self.alpha))
        self.indexes = np.random.choice(np.arange(n_elements), size=batch_size, replace=False, p=probabilities)
        w_i = np.power(1/(n_elements * probabilities[self.indexes]), self.beta)
        self.weights = w_i / np.max(w_i)


    def sample(self, batch_size):
        self.compute_indexes(batch_size)
        states, actions, rewards, next_states, dones = zip(*np.array(self.buffer)[self.indexes])
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
 

    def update_prior_buffer(self, priorities):
        temp_priorities = np.array(self.priorities)
        temp_priorities[self.indexes] = priorities.cpu().detach().numpy().astype('float').flatten()
        self.priorities = deque(temp_priorities, maxlen=self.capacity)


    def update_beta(self, episode):
        beta =  self.beta_start + (1 - self.beta_start)/self.end_beta_incr*episode
        self.beta = beta if beta < 1 else 1