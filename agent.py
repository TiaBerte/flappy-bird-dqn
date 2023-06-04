import torch 
import torch.optim as optim
import numpy as np
import random
from network import DQNetwork, DuelingDQNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplay


class Agent:
    def __init__(self, 
                 state_dim : int, 
                 n_actions : int, 
                 dueling : bool, 
                 hidden_dim : list = [256, 256], 
                 buffer_size : int = 100000, 
                 batch_size : int = 256, 
                 gamma : float = 0.99, 
                 tau : float = 0.005, 
                 lr : float = 5e-6, 
                 max_eps : float = 0.1, 
                 min_eps : float = 0.01, 
                 decay_ep : int = 2250, 
                 per : bool = False, 
                 alpha : float = 0.6, 
                 beta_start : float = 0.4, 
                 end_beta_incr : int = 2250):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.version = "DQN" 
        if dueling:
            self.version = "Dueling" + self.version
        if per:
            self.version = self.version + "_PER"

        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.decay_ep = decay_ep
        self.eps = max_eps
        self.per = per
        self.batch_size = batch_size
        
        self.buffer = ReplayBuffer(buffer_size) if not per else PrioritizedReplay(buffer_size, alpha, beta_start, end_beta_incr)

        self.q_net = DQNetwork(state_dim, n_actions, hidden_dim) if not dueling else DuelingDQNetwork(state_dim, n_actions, hidden_dim)
        self.q_net.to(self.device)

        self.q_net_target = DQNetwork(state_dim, n_actions, hidden_dim) if not dueling else DuelingDQNetwork(state_dim, n_actions, hidden_dim)
        self.q_net_target.to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    
    def act(self, 
            state : torch.tensor, 
            eval_mode : bool = False):
        random_action = np.random.choice([True, False], p=[self.eps, 1-self.eps])

        if not random_action or eval_mode:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.q_net(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.n_actions)
        return action

    def update_model(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        pred_q_val = torch.gather(self.q_net(state), dim=1, index=action.unsqueeze(1))
        target_q_val = self.gamma * torch.max(self.q_net_target(next_state), dim=1, keepdim=True)[0] *(1 - done) + reward
        td_error = torch.abs(pred_q_val - target_q_val)
        if self.per:
            self.buffer.update_prior_buffer(td_error)
            td_error = torch.tensor(self.buffer.weights) * td_error
        loss = torch.mean(td_error**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        
    def update_target(self):
        for t_p, p in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            updated_param = t_p.data * (1.0 - self.tau) + p.data * self.tau
            t_p.data.copy_(updated_param)

    def update_epsilon(self, curr_ep : int):
        self.eps = self.max_eps + (self.min_eps - self.max_eps)/self.decay_ep*curr_ep if curr_ep < self.decay_ep else self.min_eps

    def save_model(self, path : str):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path : str):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net_target.load_state_dict(torch.load(path))
 