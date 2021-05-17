import torch
import torch.nn as nn
import torch.functional as F

import numpy as np

import gym
import time

from rollout_buffer import RolloutBuffer

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

# class RolloutBuffer():
#     def __init__(self,N_STEPS=512, batch_size = 32, action_dim=2, state_dim=0):
#         self.computed_values = False
#         self.n_items = N_STEPS//batch_size
#         self.states = np.zeros((N_STEPS, state_dim), dtype=np.float32)
#         self.actions = np.zeros((N_STEPS, ))
#         self.log_probs = np.zeros((N_STEPS, ), dtype=np.float32)
#         self.rewards = np.zeros((N_STEPS, ), dtype=np.float32)
#         self.dones = np.zeros((N_STEPS, ))
#         self.values = np.zeros((N_STEPS, ), dtype=np.float32)
#         self.idx = 0 # Index of starting of batch
#         # self.length = 0
#         self.batch_size = batch_size
#     def add(self, state, action, reward, done, log_prob):
#         idx = self.idx
#         # if idx >= 512:
#         #     return
#         # batch_size = self.batch_size
#         self.states[idx] = state.copy()
#         self.actions[idx] = action.copy()
#         self.rewards[idx] = (reward)
#         self.dones[idx] = done
#         self.log_probs[idx] = log_prob
#         self.computed_values = False
#         self.idx += 1
#     def iterator(self, batch_size=64):
#         self.compute_values()
#         idx = 0
#         while idx + batch_size < self.size:
#             yield self.states[idx:idx+batch_size],self.actions[idx:idx+batch_size],self.values[idx:idx+batch_size],self.log_probs[idx:idx+batch_size]
#             idx += batch_size
#         idx = 0
#         self.idx = 0
#     def compute_values(self, last_value=0,gamma=0.99):
#         # print(len(self.actions))
#         n = self.idx
#         # print("n",n)
#         # self.values = []
#         running_sum = last_value
#         for i in range(n-1,-1,-1):
#             if self.dones[i]:
#                 running_sum = self.rewards[i]
#             else:
#                 running_sum = self.rewards[i] + gamma*running_sum
#             self.values[i] =  running_sum
#         # self.compute_values = True
#     def clear(self):
#         # self.states = []
#         # self.actions = []
#         # self.rewards = []
#         # self.dones = []
#         # self.values = []
#         # self.log_probs = []
#         self.idx = 0

#     def __iter__(self):
#         self.idx = 0
#         return self
#     def __next__(self):
#         idx, batch_size = self.idx, self.batch_size
#         if self.idx + self.batch_size < len(self.states):
#             s,a,v,l = self.states[idx:idx+batch_size],self.actions[idx:idx+batch_size],self.values[idx:idx+batch_size],self.log_probs[idx:idx+batch_size]
#             self.idx+=1
#             return s,a,v,l
#         else:
#             raise StopIteration


buffer = RolloutBuffer(256, 128,2,4)
env = gym.make('CartPole-v1')

print(env.action_space)
print(env.observation_space)

N_TIMESTEPS = int(1e5)
N_ITERATION_TIMESTEPS = 256
N_EPOCHS = 10

actor = Actor().cuda()
critic = Critic().cuda()

# torch.load()

total_timesteps = 0

opt = torch.optim.Adam(actor.parameters(), lr=0.001)
opt_critic = torch.optim.Adam(critic.parameters(), lr=0.001)
episodes_passed = 0
iteration = 0
while total_timesteps < N_TIMESTEPS:
    iteration_timesteps = 0
    state = env.reset()
    
    tstart = time.time()
    buffer.clear()
    while iteration_timesteps < N_ITERATION_TIMESTEPS:
        state = state[None,:]
        state = torch.tensor(state).float().cuda()

        action_params = actor(state)
        action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
        log_prob = torch.distributions.Categorical(logits=action_params[0]).log_prob(action).detach().item()
        # print(log_prob)
        action = action[0].detach().cpu().numpy()
        next_state, reward, done, info = env.step(action)

        # ts2 = time.time()
        buffer.add(state[0].detach().cpu().numpy(), action, reward, done, log_prob)
        # print(time.time()-ts2)
        
        if done:
            next_state = env.reset()
            episodes_passed += 1
        state = next_state

        iteration_timesteps += 1
        total_timesteps += 1
    state = state[None,:]
    state = torch.tensor(state).float().cuda()
    last_value = critic(state)[0].detach().cpu().numpy().item()
    # print(last_value)
    buffer.compute_values(last_value)
    print("Collection time", time.time()-tstart)
    for epoch in range(N_EPOCHS):
        for states, actions, values, old_log_prob in buffer:
            # print(states)
            # actions = np.array(actions).float()
            old_log_prob = np.array(old_log_prob)
            actions = torch.from_numpy(actions).long().cuda()
            # actions = torch.tensor(actions)
            states = torch.tensor(states)
            values = torch.tensor(values).flatten()
            old_log_prob = torch.from_numpy(old_log_prob).cuda().reshape(-1,1)
            # print(old_log_prob.shape)
            # with torch.no_grad():
            #     action_params = actor(states)
            #     old_log_prob = torch.distributions.Categorical(logits=action_params).log_prob(actions)

            opt_critic.zero_grad()
            V = critic(states).flatten()
            loss = 0.5 * nn.MSELoss()(V,values)
            # print(loss)
            loss.backward()
            opt_critic.step()
            # print(V)
            V = V.detach()
            # print("V", V)
            advantages = values - V
            # print(advantages.shape)
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
            advantages = advantages.squeeze()
            # print(states.shape)
            # print(action_params.shape)
            # print(old_log_prob.shape)
            opt.zero_grad()
            action_params = actor(states)
            log_prob = torch.distributions.Categorical(logits=action_params).log_prob(actions)
            # print("log_prob", log_prob)
            ratio = torch.exp(log_prob - old_log_prob).squeeze()
            # print(ratio.shape)
            l1 = ratio*advantages
            l2 = torch.clip(ratio, 0.8,1.2)*advantages
            # print(l2)
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
        state = env.reset()
        done = False
        while not done:
            state = state[None,:]
            state = torch.tensor(state).float().cuda()

            action_params = actor(state)
            action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
            action = action[0].detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
    print(iteration,episodes_passed, total_timesteps, "avg reward", total_reward/10)
    torch.save(actor.state_dict(), "./actor_cartpole.pt")
    torch.save(actor.state_dict(), "./critic_cartpole.pt")
    state = env.reset()
    done = False
    # while not done:
    #     state = state[None,:]
    #     state = torch.tensor(state).float().cuda()

    #     action_params = actor(state)
    #     action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
    #     action = action[0].detach().cpu().numpy()
    #     env.render()
    #     next_state, reward, done, info = env.step(action)
    #     state = next_state
    # env.close()

    # total_timesteps += 1

