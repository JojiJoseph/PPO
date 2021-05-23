"""
Class PPO Algorithm
"""

from typing import Deque
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gym
import time

from rollout_buffer import RolloutBuffer


from net import ActorCritic, ActorCriticContinuous

class PPO():
    def __init__(self, actor=None, critic=None, learning_rate=1e-3, env_name="CartPole-v1",
        n_timesteps=int(1e6), batch_size=64, n_epochs=10, n_rollout_timesteps=1024, coeff_v=0.5,
        clip_range=0.2,n_eval_episodes=5, device=None, max_grad_norm = None, coeff_entropy=0.0):

        self.LEARNING_RATE = 1e-3
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

    def learn(self):

        high_score = -np.inf
        device = self.DEVICE
        print("Device: ", device)
        env = gym.make(self.ENV_NAME)

        self.env = env

        episodic_returns = Deque(maxlen=100)

        if type(env.action_space) == gym.spaces.Discrete:
            n_actions = env.action_space.n
        elif type(env.action_space) == gym.spaces.Box:
            action_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError
        state_dim = env.observation_space.shape[0]


        if type(env.action_space) == gym.spaces.Discrete:
            actor_critic = ActorCritic(state_dim, n_actions).to(device)
            self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, 1, state_dim)
        elif type(env.action_space) == gym.spaces.Box:
            actor_critic = ActorCriticContinuous(state_dim, action_dim).to(device)
            self.buffer = RolloutBuffer(self.N_ROLLOUT_TIMESTEPS, self.BATCH_SIZE, action_dim, state_dim)
        else:
            raise NotImplementedError
        
        total_timesteps = 0

        opt = torch.optim.Adam(actor_critic.parameters(), lr=self.LEARNING_RATE)

        episodes_passed = 0
        iteration = 0
        _state = env.reset() # Unconverted state
        episodic_reward = 0
        while total_timesteps < self.N_TIMESTEPS:
            
            rollout_timesteps = 0
            
            rollout_start_time = time.time()
            
            self.buffer.clear()
            

            t_train_start = time.time()
            while rollout_timesteps < self.N_ROLLOUT_TIMESTEPS:
                with torch.no_grad():
                    state = _state[None,:]
                    state = torch.as_tensor(state).float().to(device)

                    if type(env.action_space) == gym.spaces.Discrete:
                        prob_params, _ = actor_critic(state)
                        distrib = torch.distributions.Categorical(logits=prob_params[0])
                        action = distrib.sample((1,))
                        log_prob = distrib.log_prob(action).item()

                        action = action[0].cpu().numpy()
                    else:
                        prob_params, _ = actor_critic(state)
                        mu, log_sigma = prob_params
                        distrib = torch.distributions.Normal(mu[0], log_sigma.exp())
                        action = distrib.sample((1,))
                        log_prob = distrib.log_prob(action).sum(dim=1).item()

                        action = action[0].cpu().numpy()
                    next_state, reward, done, info = env.step(action)
                    episodic_reward += reward
                    self.buffer.add(_state, action, reward, done, log_prob)
                
                if done:
                    next_state = env.reset()
                    episodes_passed += 1
                    episodic_returns.append(episodic_reward)
                    episodic_reward = 0

                _state = next_state

                rollout_timesteps += 1
                total_timesteps += 1
            state = _state[None,:]
            with torch.no_grad():
                state = torch.as_tensor(state).float().to(device)
                _, last_value = actor_critic(state)
                last_value = last_value[0].cpu().numpy().item()

            self.buffer.compute_values(last_value)

            for epoch in range(self.N_EPOCHS):
                for states, actions, values, old_log_prob in self.buffer:
                    if type(env.action_space) == gym.spaces.Discrete:
                        actions = torch.as_tensor(actions).long().flatten().to(device)
                    else:
                        actions = torch.as_tensor(actions).float().to(device)

                    states = torch.as_tensor(states).to(device)
                    values = torch.as_tensor(values).flatten().to(device)
                    old_log_prob = torch.as_tensor(old_log_prob).to(device)

                    opt.zero_grad()
                    action_params, values_pred = actor_critic(states)
                    values_pred = values_pred.flatten()

                    loss_critic = self.COEFF_V * F.mse_loss(values_pred,values)
                    
                    advantages = values - values_pred.detach()
                    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                    advantages = advantages.flatten()

                    if type(env.action_space) == gym.spaces.Discrete:
                        distrib = torch.distributions.Categorical(logits=action_params)
                        log_prob = distrib.log_prob(actions)
                        # print(distrib.entropy().shape)
                        # entropy = 0
                        entropy_loss = -distrib.entropy().mean()
                    else:
                        mu, log_sigma = action_params
                        entropy = 0 # TODO
                        log_prob = torch.distributions.Normal(mu, log_sigma.exp()).log_prob(actions).sum(dim=1)

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
                    torch.save(actor_critic.state_dict(), "./" + self.ENV_NAME + ".pt")
            print("Training time = ", t_train_end - t_train_start)
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
                state = _state[None,:]
                with torch.no_grad():
                    state = torch.as_tensor(state).float().to(device)

                    action_params, _ = actor_critic(state)
                    if type(env.action_space) == gym.spaces.Discrete:
                        action = torch.argmax(torch.softmax(action_params[0],-1))
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