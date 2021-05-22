
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,n_actions)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

class Critic(nn.Module):
    def __init__(self, state_dim)  -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.actor = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)
    def forward(self, x):
        return self.actor(x), self.critic(x)