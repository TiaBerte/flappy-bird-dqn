import torch.nn as nn
import torch

class DQNetwork(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int, 
                 hidden_dim : list):
        super().__init__()
        
        dim_list = [input_dim] + hidden_dim + [n_actions]
        layers = []
        for i in range(len(dim_list)-2):
            layers += [nn.Linear(dim_list[i], dim_list[i+1])]
            layers += [nn.LeakyReLU()]
        layers += [nn.Linear(dim_list[-2], dim_list[-1])]
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x : torch.tensor):
        for layer in self.layers:
            x = layer(x)
        return x



class DuelingDQNetwork(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int, 
                 hidden_dim : int):
        super().__init__()

        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim[0]),nn.LeakyReLU())
        
        dim_list = hidden_dim + [n_actions]
        value_layers = []
        advantage_layers = []
        for i in range(len(dim_list)-2):
            value_layers += [nn.Linear(dim_list[i], dim_list[i+1]), nn.LeakyReLU()]
            advantage_layers += [nn.Linear(dim_list[i], dim_list[i+1]), nn.LeakyReLU()]
        value_layers += [nn.Linear(dim_list[-2], dim_list[-1])]
        advantage_layers += [nn.Linear(dim_list[-2], dim_list[-1])]

        self.value_layers = nn.ModuleList(value_layers)
        self.advantage_layers = nn.ModuleList(advantage_layers)

        
    def forward(self, x : torch.tensor):
        x = self.input_layer(x)
        advantage = x
        value = x
        for layer in self.advantage_layers:
            advantage = layer(advantage)
        
        for layer in self.value_layers:
            value = layer(value)

        q_values = value + advantage - torch.mean(advantage)
        
        return q_values
    
