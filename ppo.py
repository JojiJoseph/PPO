"""
Class PPO Algorithm
"""

from numpy.lib.histograms import histogram
from atari_wrapper import AtariRamWrapper, BreakoutBlindWrapper
from frame_stack_atari import AtariFrameStackWrapper
from frame_stack_wrapper import FrameStackWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym.wrappers import AtariPreprocessing
import numpy as np

from typing import Deque
import csv
import time
import os
import yaml

from rollout_buffer import RolloutBuffer
from net import ActorCritic, ActorCriticContinuous, CnnActorCriticContinuos, CnnAtari

DEBUG = False

class PPO():
    def __init__(self, learning_rate=1e-3, env_name="CartPole-v1",
        n_timesteps=int(1e6), batch_size=64, n_epochs=10, n_rollout_timesteps=1024, coeff_v=0.5,
        clip_range=0.2,n_eval_episodes=5, device=None, max_grad_norm = None, coeff_entropy=0.0,
        obs_normalization=None, obs_shift=None, obs_scale=None,rew_normalization=None, rew_shift=None, rew_scale=None,
        action_scale=1, net_size=64, namespace=None, gamma=0.99, lda=0.99, wrapper=None, policy=None,
        thresh_min_return=None, wrappers=[], adv_normalization=True, resume=False):

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
        self.GAMMA = gamma
        self.LDA = lda
        self.THRESH_MIN_RETURN = thresh_min_return
        self.WRAPPERS = wrappers
        self.POLICY = policy
        self.ADV_NORMALIZATION = adv_normalization
        self.RESUME = resume
        if namespace:
            os.makedirs("./results/" + namespace, exist_ok=True)
            self.save_dir = "./results/" + namespace

    def normalize_obs(self, observation):
        if self.OBS_NORMALIZATION == "simple":
            if self.OBS_SHIFT is not None:
                observation += self.OBS_SHIFT
            if self.OBS_SCALE is not None:
                observation /= self.OBS_SCALE
        elif self.OBS_NORMALIZATION == "welford":
            std = np.sqrt(self.welford_M2 / self.welford_count)
            observation = (observation - self.welford_mean)/ std#, -10, 10)
            observation = np.clip(observation, -10, 10)
            # observation = (observation - self.welford_mean)/self.OBS_SCALE
            # print(observation)
        return observation

    def normalize_rew(self, reward):
        if self.REW_NORMALIZATION == "simple":
            if self.REW_SHIFT is not None:
                reward += self.REW_SHIFT
            if self.REW_SCALE is not None:
                reward /= self.REW_SCALE
        elif self.REW_NORMALIZATION == "welford":
            std = np.sqrt(self.welford_ret_M2 / self.welford_count)
            reward = reward/ std#, -10, 10)
            reward = np.clip(reward, -10, 10)
        return reward

    def create_env(self):
        env = gym.make(self.ENV_NAME)
        if "frame_stack" in self.WRAPPERS:
            env = FrameStackWrapper(env)
        if "atari_ram_wrapper" in self.WRAPPERS:
            env = AtariRamWrapper(env)
            # env = AtariPreprocessing(env)
        if "atari_wrapper" in self.WRAPPERS:
            env = AtariFrameStackWrapper(AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, terminal_on_life_loss=False, scale_obs=True))
        if "breakout_blind_wrapper" in self.WRAPPERS:
            env = BreakoutBlindWrapper(AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, terminal_on_life_loss=True, scale_obs=True))
        return env

    def create_network(self):
        env = self.env
        device = self.DEVICE
        state_dim = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.Discrete:
                n_actions = env.action_space.n
                actor_critic = ActorCritic(state_dim, n_actions, self.NET_SIZE).to(device)
                self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, state_dim)
                if self.POLICY == "cnn_atari":
                    actor_critic = CnnAtari(n_actions).to(device)
                    self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, 84*84*4)
        elif type(env.action_space) == gym.spaces.Box:
            action_dim = env.action_space.shape[0]
            actor_critic = ActorCriticContinuous(state_dim, action_dim, self.ACTION_SCALE, size=self.NET_SIZE).to(device)
            self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, action_dim, state_dim)
            if self.POLICY == "cnn_car_racing":
                actor_critic = CnnActorCriticContinuos(4, action_dim).to(device)
                self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, action_dim, 96*96*4)
        else:
            raise NotImplementedError
        return actor_critic

    def welford_update(self, observation):
        self.welford_count += 1
        # print(observation.shape, self.welford_mean.shape, self.welford_M2.shape)
        delta = observation - self.welford_mean
        self.welford_mean += delta/self.welford_count
        delta2 = observation - self.welford_mean
        self.welford_M2 += delta * delta2
        # self.welford_M2 += delta*delta
    def welford_rew_update(self, ret):
        if self.OBS_NORMALIZATION != "welford":
            self.welford_count += 1
        # print(observation.shape, self.welford_mean.shape, self.welford_M2.shape)
        delta = ret - self.welford_ret_mean
        self.welford_ret_mean += delta/self.welford_count
        delta2 = ret - self.welford_ret_mean
        self.welford_ret_M2 += delta * delta2

    

    def learn(self):

        # high_score = -np.inf
        device = self.DEVICE
        print("Device: ", device)
        env = self.create_env()
        if self.NAMESPACE:
            log_filename = self.save_dir + "/result.csv"
        else:
            log_filename = "./"+self.ENV_NAME+".csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        self.env = env

        self.welford_mean = np.zeros((env.observation_space.shape[0],), np.float64)
        self.welford_M2 = np.ones((env.observation_space.shape[0],), np.float64)
        self.welford_count = 1
        self.welford_ret_mean = 0
        self.welford_ret_M2 = 1

        episodic_returns = Deque(maxlen=100)

        state_dim = env.observation_space.shape[0]

        # if type(env.action_space) == gym.spaces.Discrete:
        #     n_actions = env.action_space.n
        #     actor_critic = ActorCritic(state_dim, n_actions, self.NET_SIZE).to(device)
        #     self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, state_dim)
        # elif type(env.action_space) == gym.spaces.Box:
        #     action_dim = env.action_space.shape[0]
        #     # actor_critic = ActorCriticContinuous(state_dim, action_dim, self.ACTION_SCALE, size=self.NET_SIZE).to(device)
        #     actor_critic = CnnActorCriticContinuos(4, action_dim).to(device)
        #     self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, action_dim, 96*96*4)#state_dim)
        # else:
        #     raise NotImplementedError

        actor_critic = self.create_network()

        training_info = {}
        training_info["episodes"] = 0
        training_info["timesteps"] = 0
        training_info["iteration"] = 0
        training_info["high_score"] = -np.inf
        if self.RESUME:
            actor_critic.load_state_dict(torch.load(self.save_dir + "/checkpoint.pt"))
            with open(self.save_dir + "/progress.yaml","r") as f:
                training_info = yaml.safe_load(f)
            with open(log_filename,'r',newline='') as file:
                reader = csv.reader(file)
                log_data = []
                for row in reader:
                    log_data.append(row)
                # writer.writerows(log_data)
        

        opt = torch.optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)

        episodes_passed = training_info["episodes"]
        iteration = training_info["iteration"]
        total_timesteps = training_info["timesteps"]
        high_score = training_info["high_score"]
        

        _state = env.reset() # Unconverted state
        # print("State",_state.shape)
        episodic_reward = 0
        if DEBUG: # For debugging purpose
            min_state = [np.inf]*env.observation_space.shape[0]
            max_state = [-np.inf]*env.observation_space.shape[0]
            shift = 0
        running_ret = 0
        while total_timesteps < self.N_TIMESTEPS:
            rollout_timesteps = 0
            self.buffer.clear()
            t_train_start = time.time()
            while rollout_timesteps < self.N_ROLLOUT_TIMESTEPS:
                with torch.no_grad():
                    if DEBUG:
                        min_state = np.minimum(min_state, _state)
                        max_state = np.maximum(max_state,_state)
                    if (self.OBS_NORMALIZATION == "welford"):
                        # print("welford update")
                        self.welford_update(_state)
                        # print(self.welford_mean, self.welford_M2, self.welford_count)
                    # print("\n",_state)
                    # print(self.welford_mean)
                    _state = self.normalize_obs(_state) 
                    # print(_state)
                    state = _state[None,:]
                    # print("H", state)
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

                    running_ret = running_ret*self.GAMMA + reward 
                    if self.REW_NORMALIZATION == "welford":
                        self.welford_rew_update(running_ret)
                    reward = self.normalize_rew(reward)
                    value = value.cpu().detach().numpy()
                    if self.THRESH_MIN_RETURN and episodic_reward < self.THRESH_MIN_RETURN:
                        done = True
                    self.buffer.add(_state.flatten(), action, reward, done, log_prob, value)
                if done:
                    next_state = env.reset()
                    episodes_passed += 1
                    episodic_returns.append(episodic_reward)
                    log_data.append([episodes_passed, total_timesteps+1, episodic_reward])
                    episodic_reward = 0
                    env.close()
                    env = self.create_env()
                    env.reset()
                    running_ret = 0

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

            self.buffer.compute_values(last_value, self.GAMMA, self.LDA)

            for epoch in range(self.N_EPOCHS):
                for states, actions, advantages, values, old_log_prob in self.buffer:
                    if type(env.action_space) == gym.spaces.Discrete:
                        actions = torch.as_tensor(actions).long().flatten().to(device)
                    else:
                        actions = torch.as_tensor(actions).float().to(device)

                    states = torch.as_tensor(states).to(device)
                    if self.POLICY == "cnn_car_racing":
                        states = states.reshape(self.BATCH_SIZE, 4, 96, 96).float()
                    if self.POLICY == "cnn_atari":
                        states = states.reshape(self.BATCH_SIZE, 4, 84, 84).float()
                    values = torch.as_tensor(values).flatten().to(device)
                    old_log_prob = torch.as_tensor(old_log_prob).to(device)
                    advantages = torch.as_tensor(advantages).flatten().to(device)
                    opt.zero_grad()
                    action_params, values_pred = actor_critic(states)
                    values_pred = values_pred.flatten()

                    loss_critic = self.COEFF_V * F.mse_loss(values_pred,values)
                    if self.ADV_NORMALIZATION:
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
                        entropy_loss = -distrib.entropy().sum(dim=1).mean()

                    ratio = torch.exp(log_prob - old_log_prob).squeeze()
                    l1 = ratio*advantages
                    l2 = torch.clip(ratio, 1 - self.CLIP_RANGE, 1 + self.CLIP_RANGE)*advantages
                    loss_actor = -torch.min(l1,l2)
                    loss = loss_actor.mean() + loss_critic + self.COEFF_ENTROPY*entropy_loss
                    loss.backward()
                    if self.MAX_GRAD_NORM is not None:
                        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), self.MAX_GRAD_NORM)
                    opt.step()
                    del states
                    del loss
                    del l1
                    del l2
                    del advantages
            self.buffer.clear()

            iteration += 1
            total_reward = 0
            t_train_end = time.time()
            self.actor_critc = actor_critic
            print("\nIteration = ", iteration)
            print("Avg. Return = ", np.mean(episodic_returns))
            if iteration % 10 == 0:
                t_evaluation_start = time.time()
                evaluation_score = self.evaluate()
                t_evaluation_end = time.time()
                print("Evaluation_time = ", t_evaluation_end - t_evaluation_start)
                print("Avg. Return (evaluation) = ", evaluation_score)
                if evaluation_score >= high_score:
                    print("Saved!")
                    high_score = evaluation_score
                    if self.OBS_NORMALIZATION == "welford":
                        actor_critic.welford_mean.data = torch.tensor(self.welford_mean.copy())
                        actor_critic.welford_M2.data = torch.tensor(self.welford_M2.copy())
                        actor_critic.welford_count.data = torch.tensor(self.welford_count)
                    if self.NAMESPACE:
                        torch.save(actor_critic.state_dict(), self.save_dir + "/model.pt")
                    else:
                        torch.save(actor_critic.state_dict(), "./" + self.ENV_NAME + ".pt")
                training_info["iteration"] = iteration
                training_info["timesteps"] = total_timesteps
                training_info["episodes"] = episodes_passed
                training_info["high_score"] = high_score
                with open(self.save_dir + "/progress.yaml", "w",newline='') as f:
                    yaml.safe_dump(training_info,f)
                with open(log_filename,'w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(log_data)
                torch.save(actor_critic.state_dict(), self.save_dir + "/checkpoint.pt")
                print("Training time = ", t_train_end - t_train_start)
                with open(log_filename,'w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(log_data)
    
    def evaluate(self):
        device = self.DEVICE
        total_reward = 0
        env = self.env
        actor_critic = self.actor_critc

        env = self.create_env()
        # env = self.eval_env
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
                action = action.detach().cpu().numpy()
                next_state, reward, done, info = env.step(action)
                _state = next_state
                total_reward += reward
        env.close()
        return total_reward / self.N_EVAL_EPISODES