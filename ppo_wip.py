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

from rollout_buffer import RolloutBuffer, RolloutBufferMultiEnv
from net import *

DEBUG = False

class PPO():
    def __init__(self, learning_rate=3e-4, env_name="CartPole-v1",
        n_timesteps=int(1e6), batch_size=64, n_epochs=10, n_rollout_timesteps=2048, coeff_v=0.5,
        clip_range=0.2,n_eval_episodes=5, device=None, max_grad_norm = 0.5, coeff_entropy=0.0,
        obs_normalization=None, obs_shift=None, obs_scale=None,rew_normalization=None, rew_shift=None, rew_scale=None,
        action_scale=1, net_size=64, namespace=None, gamma=0.99, lda=0.99, wrapper=None, policy=None,
        thresh_min_return=None, wrappers=[], adv_normalization=True, resume=False, n_envs=1,
        max_normalization_update_steps = np.inf):

        # Hyperparameters
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
        self.N_ENVS = n_envs
        self.RESUME = resume 
        self.MAX_NORMALIZATION_STEPS = max_normalization_update_steps
        if namespace:
            os.makedirs("./results/" + namespace, exist_ok=True)
            self.save_dir = "./results/" + namespace
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def normalize_obs(self, observation):
        if self.OBS_NORMALIZATION == "simple":
            if self.OBS_SHIFT is not None:
                observation += self.OBS_SHIFT
            if self.OBS_SCALE is not None:
                observation /= self.OBS_SCALE
        elif self.OBS_NORMALIZATION == "welford":
            # Leave the following comment
            # std = np.sqrt(self.welford_M2 / self.welford_count + 1e-8)
            std = np.sqrt(self.welford_var)
            observation = (observation - self.welford_mean)/ std#, -10, 10)
            observation = np.clip(observation, -10, 10)
        return observation

    def normalize_rew(self, reward):
        if self.REW_NORMALIZATION == "simple":
            if self.REW_SHIFT is not None:
                reward += self.REW_SHIFT
            if self.REW_SCALE is not None:
                reward /= self.REW_SCALE
        elif self.REW_NORMALIZATION == "welford":
            # std = np.sqrt(self.welford_ret_M2 / self.welford_count + 1e-8) # Leave this comment as it is
            std = np.sqrt(self.welford_ret_var + 1e-8)
            reward = reward/ std
            reward = np.clip(reward, -10, 10)
        return reward

    def create_env(self):
        env = gym.make(self.ENV_NAME)
        if "frame_stack" in self.WRAPPERS:
            env = FrameStackWrapper(env)
        if "atari_ram_wrapper" in self.WRAPPERS:
            env = AtariRamWrapper(env)
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
                if self.POLICY == "cnn_atari":
                    actor_critic = CnnAtari(n_actions).to(device)
                if self.POLICY == "mlp2":
                    actor_critic = ActorCritic2(state_dim, n_actions, self.NET_SIZE).to(device)
                    
        elif type(env.action_space) == gym.spaces.Box:
            action_dim = env.action_space.shape[0]
            actor_critic = ActorCriticContinuous(state_dim, action_dim, self.ACTION_SCALE, size=self.NET_SIZE).to(device)
            if self.POLICY == "cnn_car_racing":
                actor_critic = CnnActorCriticContinuos(4, action_dim).to(device)
            if self.POLICY == "mlp2":
                actor_critic = ActorCriticContinuous2(state_dim, action_dim, self.ACTION_SCALE, size=self.NET_SIZE).to(device)
        else:
            raise NotImplementedError
        return actor_critic

    def welford_update(self, observation):
        b_mean = np.mean(observation, axis=0)
        b_M2 = np.var(observation, axis=0)*self.N_ENVS
        self.welford_count += self.N_ENVS
        delta = b_mean - self.welford_mean
        self.welford_mean += delta*self.N_ENVS/self.welford_count
        self.welford_M2 += b_M2 + np.square(delta) * (self.welford_count-self.N_ENVS) * self.N_ENVS/ self.welford_count

        # Test lines
        M2 = self.welford_var * (self.welford_count-self.N_ENVS) + b_M2 + np.square(delta) * (self.welford_count-self.N_ENVS) * self.N_ENVS/ self.welford_count
        self.welford_var = M2 / self.welford_count
            
    def welford_rew_update(self, ret):
        self.welford_ret_count += self.N_ENVS
        b_mean = np.mean(ret)
        b_M2 = np.var(ret)*self.N_ENVS
        delta = b_mean - self.welford_ret_mean
        self.welford_ret_mean += delta*self.N_ENVS/self.welford_ret_count
        self.welford_ret_M2 += b_M2 + np.square(delta) * (self.welford_ret_count-self.N_ENVS) * self.N_ENVS/ self.welford_ret_count

        # Test lines
        M2 = self.welford_ret_var * (self.welford_ret_count-self.N_ENVS) + b_M2 + np.square(delta) * (self.welford_ret_count-self.N_ENVS) * self.N_ENVS/ self.welford_ret_count
        self.welford_ret_var = M2 / self.welford_ret_count


    
    def create_buffer(self):
        env = self.env
        buffer = None
        if type(env.action_space) == gym.spaces.Discrete:
            buffer = RolloutBufferMultiEnv(self.N_ROLLOUT_TIMESTEPS, self.N_ENVS, self.BATCH_SIZE, 1, env.observation_space.shape[0])
            if self.POLICY == "cnn_atari":
                buffer = RolloutBufferMultiEnv(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, 84*84*4)
        elif type(env.action_space) == gym.spaces.Box:
            self.buffer = RolloutBufferMultiEnv(self.N_ROLLOUT_TIMESTEPS, self.N_ENVS, self.BATCH_SIZE, env.action_space.shape[0], env.observation_space.shape[0])
            if self.POLICY == "cnn_car_racing":
                buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, env.action_space.shape[0], 96*96*4)
        return buffer

    def learn(self):

        device = self.DEVICE
        
        print("Device: ", device)
        
        env = self.create_env()
        
        # Create vector of environments
        envs = [self.create_env() for i in range(self.N_ENVS)]
        self.envs = envs

        if self.NAMESPACE:
            log_filename = self.save_dir + "/result.csv"
        else:
            log_filename = "./"+self.ENV_NAME+".csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        self.env = env

        # Statistics of observations and returns
        self.welford_mean = np.zeros((env.observation_space.shape[0],), np.float64)
        self.welford_M2 = np.ones((env.observation_space.shape[0],), np.float64)
        self.welford_var = np.ones((env.observation_space.shape[0],), np.float64)
        self.welford_count = np.array(1e-4, dtype=np.float64)
        self.welford_ret_count = np.array(1e-4, dtype=np.float64)
        self.welford_ret_mean = np.array(0, np.float64)
        self.welford_ret_M2 = np.array(1, np.float64)
        self.welford_ret_var = np.array(1, np.float64)

        # The queue that stores last 100 episodes.
        # Used to caculate mean score for the last 100 episodes
        episodic_returns = Deque(maxlen=100)

        state_dim = env.observation_space.shape[0]

        actor_critic = self.create_network()

        # Create buffer
        self.buffer = self.create_buffer()

        # The object that helps to load checkpoints
        training_info = {}
        training_info["episodes"] = 0
        training_info["timesteps"] = 0
        training_info["iteration"] = 0
        training_info["high_score"] = -np.inf
        training_info["statistics"] = {} # TODO

        if self.RESUME:
            actor_critic.load_state_dict(torch.load(self.save_dir + "/checkpoint.pt"))
            with open(self.save_dir + "/progress.yaml","r") as f:
                training_info = yaml.safe_load(f)
            with open(log_filename,'r',newline='') as file:
                reader = csv.reader(file)
                log_data = []
                for row in reader:
                    log_data.append(row)
            if self.OBS_NORMALIZATION == "welford":
                self.welford_mean = actor_critic.welford_mean.data.detach().numpy()
                self.welford_M2 = actor_critic.welford_M2.data.detach().numpy()
                self.welford_count = actor_critic.welford_count.data.detach().numpy()

        # Optimizer. TODO: Implement learning schedule
        opt = torch.optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)

        episodes_passed = training_info["episodes"]
        iteration = training_info["iteration"]
        total_timesteps = training_info["timesteps"]
        high_score = training_info["high_score"]
                
        _state = np.array([env.reset() for env in envs])
        
        episodic_reward = 0
                
        running_ret = np.zeros(self.N_ENVS)
        
        # Training loop
        while total_timesteps < self.N_TIMESTEPS:
            rollout_timesteps = 0
            self.buffer.clear()
            t_train_start = time.time()

            # Collecting data
            while rollout_timesteps < self.N_ROLLOUT_TIMESTEPS:
                with torch.no_grad():

                    if (self.OBS_NORMALIZATION == "welford") and self.welford_count < self.MAX_NORMALIZATION_STEPS:
                        self.welford_update(_state)

                    _state = self.normalize_obs(_state) 


                    state = _state
                    state = torch.as_tensor(state).float().to(device)

                    if type(env.action_space) == gym.spaces.Discrete:
                        prob_params, value = actor_critic(state)

                        distrib = torch.distributions.Categorical(logits=prob_params)
                        action = distrib.sample((1,)).flatten()

                        log_prob = distrib.log_prob(action)
                        action = action.cpu().numpy()

                    else:
                        # print("s",state.shape)
                        prob_params, value = actor_critic(state)
                        mu, log_sigma = prob_params

                        distrib = torch.distributions.Normal(mu, log_sigma.exp())
                        action = distrib.sample((1,))[0]

                        log_prob = distrib.log_prob(action).sum(dim=1)
                        action = action.cpu().numpy()
                        action = np.clip(action, -self.ACTION_SCALE, self.ACTION_SCALE)

                    batch_result = [env.step(a) for env, a in zip(envs,action)]

                    next_state, reward, done, info = [], [], [], []
                    for n, r, d, i in batch_result:
                        next_state.append(n)
                        reward.append(r)
                        done.append(d)
                        info.append(i)

                    reward = np.array(reward)
                    next_state = np.array(next_state)

                    done = np.array(done)
                    info = np.array(info)
                    episodic_reward += reward
                    running_ret = running_ret*self.GAMMA + reward 
                    if self.REW_NORMALIZATION == "welford"  and self.welford_ret_count < self.MAX_NORMALIZATION_STEPS:
                        self.welford_rew_update(running_ret)
                    reward = self.normalize_rew(reward)

                    value = value.cpu().detach().numpy()
                    if self.THRESH_MIN_RETURN and episodic_reward < self.THRESH_MIN_RETURN:
                        done = True

                    self.buffer.add(_state.reshape((self.N_ENVS,-1)), action.reshape(self.N_ENVS,-1), reward, done, log_prob.cpu(), value.reshape((self.N_ENVS,)))


                for i, d in enumerate(done):
                    if d:
                        next_state[i] = envs[i].reset()
                        episodes_passed += 1
                        episodic_returns.append(episodic_reward[i])
                        log_data.append([episodes_passed, total_timesteps+1+i, episodic_reward[i]])
                        episodic_reward[i] = 0
                        running_ret[i] = 0


                _state = next_state

                rollout_timesteps += 1 #self.N_ENVS
                total_timesteps += self.N_ENVS


            state = _state
            with torch.no_grad():
                state = self.normalize_obs(state)#.float()
                state = torch.as_tensor(state).float().to(device)
                _, last_value = actor_critic(state)

                last_value = last_value.cpu().numpy()

            self.buffer.compute_values(last_value, self.GAMMA, self.LDA)

            for epoch in range(self.N_EPOCHS):

                for id, (states, actions, advantages, values, old_log_prob) in enumerate(self.buffer):

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

                    loss_critic = self.COEFF_V * F.mse_loss(values, values_pred)
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

            self.buffer.clear()

            iteration += 1
            total_reward = 0
            t_train_end = time.time()
            self.actor_critc = actor_critic
            print("\nIteration = ", iteration)
            print("Avg. Return = ", np.mean(episodic_returns))
            print(self.welford_mean)
            print("Total timesteps = ", total_timesteps)

            if iteration % 10 == 0:
                t_evaluation_start = time.time()
                evaluation_score = self.evaluate()
                t_evaluation_end = time.time()
                print("Evaluation_time = ", t_evaluation_end - t_evaluation_start)
                print("Avg. Return (evaluation) = ", evaluation_score)
                if evaluation_score >= high_score:
                    print("Saved!")
                    print(type(high_score))
                    high_score = evaluation_score
                    high_score = np.float32(high_score).item()
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
                    print(training_info)
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
        actor_critic = self.actor_critc

        env = self.create_env()

        for episode in range(self.N_EVAL_EPISODES):
            _state = env.reset()
            done = False
            while not done:

                state = _state[None,:]
                state = self.normalize_obs(state)
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
                np.clip(action, -self.ACTION_SCALE, self.ACTION_SCALE)
                next_state, reward, done, info = env.step(action)
                _state = next_state
                total_reward += reward
        env.close()
        return total_reward / self.N_EVAL_EPISODES