# from ppo_pendulum import BATCH_SIZE
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gym
import time

from rollout_buffer import RolloutBuffer

# torch.set_default_tensor_type('torch#.cuda.FloatTensor')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
print("Using device:", device)

class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,2)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y
        
env = gym.make("CartPole-v1")

print(env.action_space)
print(env.observation_space)

action_dim = 1
state_dim = 4

# HYPER PARAMETERS
N_TIMESTEPS = int(1e6)
N_ROLLOUT_TIMESTEPS = 256
N_EPOCHS = 16
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
Cv = 0.5

buffer = RolloutBuffer(N_ROLLOUT_TIMESTEPS, BATCH_SIZE, action_dim, state_dim)

actor = Actor().to(device)
critic = Critic().to(device)

print(actor)
print(critic)

actor.load_state_dict(torch.load("./actor_cartpole.pt"))
critic.load_state_dict(torch.load("./critic_cartpole.pt"))

total_timesteps = 0

opt = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
opt_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
episodes_passed = 0
iteration = 0
while total_timesteps < N_TIMESTEPS:
    
    rollout_timesteps = 0
    
    rollout_start_time = time.time()
    
    buffer.clear()
    
    _state = env.reset() # Unconverted state
    
    while rollout_timesteps < N_ROLLOUT_TIMESTEPS:
        with torch.no_grad():
            state = _state[None,:]
            state = torch.as_tensor(state).float().to(device)

            prob_params = actor(state)
            distrib = torch.distributions.Categorical(logits=prob_params[0])
            action = distrib.sample((1,))
            log_prob = distrib.log_prob(action).item()

            action = action[0].cpu().numpy()
            next_state, reward, done, info = env.step(action)
            buffer.add(_state, action, reward, done, log_prob)
        
        if done:
            next_state = env.reset()
            episodes_passed += 1
        _state = next_state

        rollout_timesteps += 1
        total_timesteps += 1
    state = _state[None,:]
    with torch.no_grad():
        state = torch.as_tensor(state).float().to(device)
        last_value = critic(state)[0].cpu().numpy().item()

    buffer.compute_values(last_value)
    print("Collection time", time.time()-rollout_start_time)
    for epoch in range(N_EPOCHS):
        for states, actions, values, old_log_prob in buffer:

            actions = torch.as_tensor(actions).long().flatten().to(device)

            states = torch.as_tensor(states).to(device)
            values = torch.as_tensor(values).flatten().to(device)
            old_log_prob = torch.as_tensor(old_log_prob).to(device)
            opt_critic.zero_grad()
            V = critic(states).flatten()
            loss = 0.5 * F.mse_loss(V,values)
            loss.backward()
            opt_critic.step()

            V = V.detach()

            advantages = values - V
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
            advantages = advantages.squeeze()

            opt.zero_grad()
            action_params = actor(states)
            log_prob = torch.distributions.Categorical(logits=action_params).log_prob(actions)
            ratio = torch.exp(log_prob - old_log_prob).squeeze()
            l1 = ratio*advantages
            l2 = torch.clip(ratio, 0.8,1.2)*advantages
            loss = -torch.min(l1,l2)
            loss = loss.mean()
            # print(loss)
            loss.backward()
            opt.step()
    buffer.clear()
    # Evaluation step
    iteration += 1
    total_reward = 0
    for episode in range(10):
        _state = env.reset()
        done = False
        while not done:
            state = _state[None,:]
            with torch.no_grad():
                state = torch.as_tensor(state).float().to(device)

                action_params = actor(state)
                action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))[0]
                # action = torch.argmax(torch.softmax(action_params[0],-1))
                action = action.cpu().numpy()
            next_state, reward, done, info = env.step(action)
            _state = next_state
            total_reward += reward
    print(iteration,episodes_passed, total_timesteps, "avg reward", total_reward/10)
    torch.save(actor.state_dict(), "./actor_cartpole.pt")
    torch.save(critic.state_dict(), "./critic_cartpole.pt")
    print(time.time()-rollout_start_time)
    # state = env.reset()
    # done = False
    # while not done:
    #     state = state[None,:]
    #     state = torch.tensor(state).float()#.cuda()

    #     action_params = actor(state)
    #     action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
    #     action = action[0].detach().cpu().numpy()
    #     env.render()
    #     time.sleep(1/30.)
    #     next_state, reward, done, info = env.step(action)
    #     state = next_state
    # env.close()

    # total_timesteps += 1

