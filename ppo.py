"""
Class PPO Algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from typing import Deque
import csv
import time
import os

from rollout_buffer import RolloutBuffer
from net import ActorCritic, ActorCriticContinuous

DEBUG = False

class PPO():
    def __init__(self, learning_rate=1e-3, env_name="CartPole-v1",
        n_timesteps=int(1e6), batch_size=64, n_epochs=10, n_rollout_timesteps=1024, coeff_v=0.5,
        clip_range=0.2,n_eval_episodes=5, device=None, max_grad_norm = None, coeff_entropy=0.0,
        obs_normalization=None, obs_shift=None, obs_scale=None,rew_normalization=None, rew_shift=None, rew_scale=None,
        action_scale=1, net_size=64, namespace=None):

        self.LEARNING_RATE = learning_rate
        self.ENV_NAME = env_name
        self.N_TIMESTEPS = n_timesteps
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs
        self.N_ROLLOUT_TIMESTEPS = n_rollout_timesteps
        self.COEFF_V = coeff_v
        self.CLIP_RANGE = clip_range
        self.N_EVAL_EPISODES = n_eval_episodes
        self.MAX_GRAD_NORM = max_grad_norm
        self.COEFF_ENTROPY = coeff_entropy
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.DEVICE = device
        self.OBS_NORMALIZATION = obs_normalization
        self.OBS_SHIFT = obs_shift
        self.OBS_SCALE = obs_scale
        self.REW_NORMALIZATION = rew_normalization
        self.REW_SHIFT = rew_shift
        self.REW_SCALE = rew_scale
        self.ACTION_SCALE = action_scale
        self.NET_SIZE = net_size
        self.NAMESPACE = namespace
        if namespace:
            os.makedirs("./results/" + namespace, exist_ok=True)
            self.save_dir = "./results/" + namespace

    def normalize_obs(self, observation):
        if self.OBS_NORMALIZATION == "simple":
            if self.OBS_SHIFT is not None:
                observation += self.OBS_SHIFT
            if self.OBS_SCALE is not None:
                observation /= self.OBS_SCALE
        return observation

    def normalize_rew(self, reward):
        if self.REW_NORMALIZATION == "simple":
            if self.REW_SHIFT is not None:
                reward += self.REW_SHIFT
            if self.REW_SCALE is not None:
                reward /= self.REW_SCALE
        return reward

    def learn(self):

        high_score = -np.inf
        device = self.DEVICE
        print("Device: ", device)
        env = gym.make(self.ENV_NAME)
        if self.NAMESPACE:
            log_filename = self.save_dir + "/result.csv"
        else:
            log_filename = "./"+self.ENV_NAME+".csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        self.env = env

        episodic_returns = Deque(maxlen=100)

        state_dim = env.observation_space.shape[0]

        if type(env.action_space) == gym.spaces.Discrete:
            n_actions = env.action_space.n
            actor_critic = ActorCritic(state_dim, n_actions, self.NET_SIZE).to(device)
            self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, state_dim)
        elif type(env.action_space) == gym.spaces.Box:
            action_dim = env.action_space.shape[0]
            actor_critic = ActorCriticContinuous(state_dim, action_dim, self.ACTION_SCALE, size=self.NET_SIZE).to(device)
            self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, action_dim, state_dim)
        else:
            raise NotImplementedError
        
        total_timesteps = 0

        opt = torch.optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)

        episodes_passed = 0
        iteration = 0
        _state = env.reset() # Unconverted state
        episodic_reward = 0
        if DEBUG: # For debugging purpose
            min_state = [np.inf]*env.observation_space.shape[0]
            max_state = [-np.inf]*env.observation_space.shape[0]
            shift = 0
        while total_timesteps < self.N_TIMESTEPS:
            rollout_timesteps = 0
            self.buffer.clear()
            t_train_start = time.time()
            while rollout_timesteps < self.N_ROLLOUT_TIMESTEPS:
                with torch.no_grad():
                    if DEBUG:
                        min_state = np.minimum(min_state, _state)
                        max_state = np.maximum(max_state,_state)
                    _state = self.normalize_obs(_state) 
                    state = _state[None,:]
                    state = torch.as_tensor(state).float().to(device)

                    if type(env.action_space) == gym.spaces.Discrete:
                        prob_params, value = actor_critic(state)
                        distrib = torch.distributions.Categorical(logits=prob_params[0])
                        action = distrib.sample((1,))
                        log_prob = distrib.log_prob(action).item()

                        action = action[0].cpu().numpy()
                    else:
                        prob_params, value = actor_critic(state)
                        mu, log_sigma = prob_params
                        distrib = torch.distributions.Normal(mu[0], log_sigma.exp())
                        action = distrib.sample((1,))
                        log_prob = distrib.log_prob(action).sum(dim=1).item()

                        action = action[0].cpu().numpy()
                    next_state, reward, done, info = env.step(action)

                    episodic_reward += reward

                    reward = self.normalize_rew(reward)
                    self.buffer.add(_state, action, reward, done, log_prob, value.detach().numpy())
                
                if done:
                    next_state = env.reset()
                    episodes_passed += 1
                    episodic_returns.append(episodic_reward)
                    log_data.append([episodes_passed, total_timesteps+1, episodic_reward])
                    episodic_reward = 0

                _state = next_state

                rollout_timesteps += 1
                total_timesteps += 1
            if DEBUG:
                print(min_state)
                print(max_state)
                shift=- (max_state + min_state)/2
                print("shift", shift)
                print("scale", abs(max_state + shift))
            state = _state[None,:]
            with torch.no_grad():
                state = torch.as_tensor(state).float().to(device)
                _, last_value = actor_critic(state)
                last_value = last_value[0].cpu().numpy().item()

            self.buffer.compute_values(last_value, 0.99, 0.99)

            for epoch in range(self.N_EPOCHS):
                for states, actions, advantages, values, old_log_prob in self.buffer:
                    if type(env.action_space) == gym.spaces.Discrete:
                        actions = torch.as_tensor(actions).long().flatten().to(device)
                    else:
                        actions = torch.as_tensor(actions).float().to(device)

                    states = torch.as_tensor(states).to(device)
                    values = torch.as_tensor(values).flatten().to(device)
                    old_log_prob = torch.as_tensor(old_log_prob).to(device)
                    advantages = torch.as_tensor(advantages).flatten().to(device)
                    opt.zero_grad()
                    action_params, values_pred = actor_critic(states)
                    values_pred = values_pred.flatten()

                    loss_critic = self.COEFF_V * F.mse_loss(values_pred,values)
                    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                    advantages = advantages.flatten()

                    if type(env.action_space) == gym.spaces.Discrete:
                        distrib = torch.distributions.Categorical(logits=action_params)
                        log_prob = distrib.log_prob(actions)
                        entropy_loss = -distrib.entropy().mean()
                    else:
                        mu, log_sigma = action_params
                        distrib = torch.distributions.Normal(mu, log_sigma.exp())
                        log_prob = distrib.log_prob(actions).sum(dim=1)
                        entropy_loss = distrib.entropy().sum(dim=1).mean()

                    ratio = torch.exp(log_prob - old_log_prob).squeeze()
                    l1 = ratio*advantages
                    l2 = torch.clip(ratio, 1 - self.CLIP_RANGE, 1 + self.CLIP_RANGE)*advantages
                    loss_actor = -torch.min(l1,l2)
                    loss = loss_actor.mean() + loss_critic + self.COEFF_ENTROPY*entropy_loss
                    loss.backward()
                    if self.MAX_GRAD_NORM is not None:
                        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), self.MAX_GRAD_NORM)
                    opt.step()
            self.buffer.clear()

            iteration += 1
            total_reward = 0
            t_train_end = time.time()
            self.actor_critc = actor_critic
            print("\nIteration = ", iteration)
            print("Avg. Return = ", np.mean(episodic_returns))
            if iteration % 10 == 1:
                t_evaluation_start = time.time()
                evaluation_score = self.evaluate()
                t_evaluation_end = time.time()
                print("Evaluation_time = ", t_evaluation_end - t_evaluation_start)
                print("Avg. Return (evaluation) = ", evaluation_score)
                if evaluation_score > high_score:
                    print("Saved!")
                    high_score = evaluation_score
                    if self.NAMESPACE:
                        torch.save(actor_critic.state_dict(), self.save_dir + "/model.pt")
                    else:
                        torch.save(actor_critic.state_dict(), "./" + self.ENV_NAME + ".pt")
            with open(log_filename,'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows(log_data)
            print("Training time = ", t_train_end - t_train_start)
        with open(log_filename,'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows(log_data)
    
    def evaluate(self):
        device = self.DEVICE
        total_reward = 0
        env = self.env
        actor_critic = self.actor_critc
        env = gym.make(self.ENV_NAME) # Eval env
        for episode in range(self.N_EVAL_EPISODES):
            _state = env.reset()
            done = False
            while not done:
                _state = self.normalize_obs(_state)
                state = _state[None,:]
                with torch.no_grad():
                    state = torch.as_tensor(state).float().to(device)

                    action_params, _ = actor_critic(state)
                    if type(env.action_space) == gym.spaces.Discrete:
                        action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))[0]
                    else:
                        mu, log_sigma = action_params
                        distrib = torch.distributions.Normal(mu[0], log_sigma.exp())
                        action = distrib.sample((1,))[0]
                    action = action.cpu().numpy()
                next_state, reward, done, info = env.step(action)
                _state = next_state
                total_reward += reward
        env.close()
        return total_reward / self.N_EVAL_EPISODES