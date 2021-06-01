
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, size=64) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size,size)
        self.l3 = nn.Linear(size,n_actions)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

class Critic(nn.Module):
    def __init__(self, state_dim, size=64)  -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size,size)
        self.l3 = nn.Linear(size,1)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

class ActorContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1,size=64) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size,size)
        self.mu = nn.Linear(size,action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.action_scale = action_scale
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        mu = torch.tanh(self.mu(y))
        # log_std = self.log_std(y)
        return self.action_scale * mu, self.log_std


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, size=64):
        super().__init__()
        self.actor = Actor(state_dim, n_actions, size)
        self.critic = Critic(state_dim, size)
    def forward(self, x):
        return self.actor(x), self.critic(x)

class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1, size=64):
        super().__init__()
        self.actor = ActorContinuous(state_dim, action_dim, action_scale,size)
        self.critic = Critic(state_dim,size)
    def forward(self, x):
        return self.actor(x), self.critic(x)

class CnnActorCriticContinuos(nn.Module):
    def __init__(self, channels, action_dim, action_scale=1, size=64):
        super().__init__()
        self.c1 = nn.Conv2d(channels, 24,kernel_size=5, stride=2)
        self.c2 = nn.Conv2d(24, 8, kernel_size=5, stride=2)
        self.l1 = nn.Linear(3528, size)
        self.l2 = nn.Linear(size, size)
        self.critic_out = nn.Linear(size, 1)
        self.l3 = nn.Linear(3528, size)
        self.l4 = nn.Linear(size, size)
        self.actor_mu = nn.Linear(size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.action_scale = action_scale
    def forward(self, x):
        # print(x.shape)
        y = torch.relu(self.c1(x))
        # print(y.shape)
        y = torch.relu(self.c2(y))
        y = y.reshape(y.shape[0], -1)
        # print(y.shape)
        critic_in = torch.relu(self.l1(y))
        critic_in = torch.relu(self.l2(critic_in))
        critic_out = self.critic_out(critic_in)
        actor_in = torch.relu(self.l3(y))
        actor_in = torch.relu(self.l4(actor_in))
        actor_mu = torch.tanh(self.actor_mu(actor_in))
        return (actor_mu, self.actor_log_std), critic_out