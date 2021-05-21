"""
Class PPO Algorithm
"""

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
device = "cpu"
print("Using device:", device)

# class Actor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.l1 = nn.Linear(4, 64)
#         self.l2 = nn.Linear(64,64)
#         self.l3 = nn.Linear(64,2)
#     def forward(self, x):
#         y = torch.relu(self.l1(x))
#         y = torch.relu(self.l2(y))
#         y = self.l3(y)
#         return y

# class Critic(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.l1 = nn.Linear(4, 64)
#         self.l2 = nn.Linear(64,64)
#         self.l3 = nn.Linear(64,1)
#     def forward(self, x):
#         y = torch.relu(self.l1(x))
#         y = torch.relu(self.l2(y))
#         y = self.l3(y)
#         return y

# class ActorCritic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.actor = Actor()
#         self.critic = Critic()
#     def forward(self, x):
#         return self.actor(x), self.critic(x)

from net import ActorCritic

class PPO():
    def __init__(self, actor=None, critic=None, learning_rate=1e-3, env_name="CartPole-v1",
        n_timesteps=int(1e6), batch_size=64, n_epochs=10, n_rollout_timesteps=1024, coeff_v=0.5,
        clip_range=0.2,n_eval_episodes=5, device=None):

        self.LEARNING_RATE = 1e-3
        self.ENV_NAME = env_name
        self.N_TIMESTEPS = n_timesteps
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs
        self.N_ROLLOUT_TIMESTEPS = n_rollout_timesteps
        self.COEFF_V = coeff_v
        self.CLIP_RANGE = clip_range
        self.N_EVAL_EPISODES = n_eval_episodes
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.DEVICE = device

    def learn(self):

        device = self.DEVICE
        print("Device: ", device)
        env = gym.make(self.ENV_NAME)

        self.env = env

        # print(env.action_space)
        # print(env.observation_space)

        # action_dim = 1
        # state_dim = 4


        # actor = Actor().to(device)
        # critic = Critic().to(device)

        n_actions = env.action_space.n
        state_dim = env.observation_space.shape[0]
        # print(state_dim, n_actions)
        actor_critic = ActorCritic(state_dim, n_actions).to(device)
        # actor_critic = ActorCritic()/
        self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, state_dim)

        # self.actor = actor
        # self.critic = critic

        # print(actor)
        # print(critic)

        # actor.load_state_dict(torch.load("./actor_cartpole.pt"))
        # critic.load_state_dict(torch.load("./critic_cartpole.pt"))
        total_timesteps = 0

        opt = torch.optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)
        # opt_critic = torch.optim.Adam(critic.parameters(), lr=self.LEARNING_RATE)
        episodes_passed = 0
        iteration = 0
        while total_timesteps < self.N_TIMESTEPS:
            
            rollout_timesteps = 0
            
            rollout_start_time = time.time()
            
            self.buffer.clear()
            
            _state = env.reset() # Unconverted state

            t_train_start = time.time()
            while rollout_timesteps < self.N_ROLLOUT_TIMESTEPS:
                with torch.no_grad():
                    state = _state[None,:]
                    state = torch.as_tensor(state).float().to(device)

                    prob_params, _ = actor_critic(state)
                    distrib = torch.distributions.Categorical(logits=prob_params[0])
                    action = distrib.sample((1,))
                    log_prob = distrib.log_prob(action).item()

                    action = action[0].cpu().numpy()
                    next_state, reward, done, info = env.step(action)
                    self.buffer.add(_state, action, reward, done, log_prob)
                
                if done:
                    next_state = env.reset()
                    episodes_passed += 1
                _state = next_state

                rollout_timesteps += 1
                total_timesteps += 1
            state = _state[None,:]
            with torch.no_grad():
                state = torch.as_tensor(state).float().to(device)
                _, last_value = actor_critic(state)
                last_value = last_value[0].cpu().numpy().item()

            self.buffer.compute_values(last_value)
            # print("Collection time", time.time()-rollout_start_time)
            for epoch in range(self.N_EPOCHS):
                for states, actions, values, old_log_prob in self.buffer:

                    actions = torch.as_tensor(actions).long().flatten().to(device)

                    states = torch.as_tensor(states).to(device)
                    values = torch.as_tensor(values).flatten().to(device)
                    old_log_prob = torch.as_tensor(old_log_prob).to(device)

                    opt.zero_grad()
                    action_params, V = actor_critic(states)
                    V = V.flatten()

                    loss_critic = self.COEFF_V * F.mse_loss(V,values)
                    
                    advantages = values - V.detach()
                    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                    advantages = advantages.flatten()
                    # print(action_params.shape)
                    # print(actions.shape)
                    log_prob = torch.distributions.Categorical(logits=action_params).log_prob(actions)

                    ratio = torch.exp(log_prob - old_log_prob).squeeze()
                    l1 = ratio*advantages
                    l2 = torch.clip(ratio, 1 - self.CLIP_RANGE, 1 + self.CLIP_RANGE)*advantages
                    loss_actor = -torch.min(l1,l2)
                    loss = loss_actor.mean() + loss_critic
                    loss.backward()
                    opt.step()
            self.buffer.clear()
            # Evaluation step
            iteration += 1
            total_reward = 0
            t_train_end = time.time()
            self.actor_critc = actor_critic
            print("\nIteration = ", iteration)
            if iteration % 10 == 1:
                t_evaluation_start = time.time()
                evaluation_score = self.evaluate()
                t_evaluation_end = time.time()
                print("evaluation_time = ", t_evaluation_end - t_evaluation_start)
                print("Avg. Return - evaluation = ", evaluation_score)
                torch.save(actor_critic.state_dict(), "./" + self.ENV_NAME + ".pt")
            print("Training time = ", t_train_end - t_train_start)
    def evaluate(self):
        device = self.DEVICE
        total_reward = 0
        env = self.env
        actor_critic = self.actor_critc
        for episode in range(self.N_EVAL_EPISODES):
            _state = env.reset()
            done = False
            while not done:
                state = _state[None,:]
                with torch.no_grad():
                    state = torch.as_tensor(state).float().to(device)

                    action_params, _ = actor_critic(state)
                    action = torch.argmax(torch.softmax(action_params[0],-1))
                    # action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))[0]
                    action = action.cpu().numpy()
                next_state, reward, done, info = env.step(action)
                _state = next_state
                total_reward += reward
        return total_reward / self.N_EVAL_EPISODES
        print(iteration,episodes_passed, total_timesteps, "avg reward", total_reward/self.N_EVAL_EPISODES)
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