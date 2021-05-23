
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

class ActorContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64,64)
        self.mu = nn.Linear(64,action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.action_scale = action_scale
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        mu = torch.tanh(self.mu(y))
        # log_std = self.log_std(y)
        return self.action_scale * mu, self.log_std


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.actor = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)
    def forward(self, x):
        return self.actor(x), self.critic(x)

class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1):
        super().__init__()
        self.actor = ActorContinuous(state_dim, action_dim, action_scale)
        self.critic = Critic(state_dim)
    def forward(self, x):
        return self.actor(x), self.critic(x)