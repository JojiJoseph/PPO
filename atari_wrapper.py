
import gym
import numpy as np
from typing import Deque

class AtariRamWrapper:
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    def reset(self):
        obs = self.env.reset().astype(np.float32)
        return obs#/255.
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.float32)#/255.
        return obs, reward, done, info
    def render(self):
        return self.env.render()
    def close(self):
        self.env.close()
