
import gym
import numpy as np
from typing import Deque
import cv2

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

class BreakoutBlindWrapper:
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.queue = Deque(maxlen=4)
        self.queue.append(np.zeros([84,84], np.float32))
        self.queue.append(np.zeros([84,84], np.float32))
        self.queue.append(np.zeros([84,84], np.float32))
        self.queue.append(np.zeros([84,84], np.float32))
    def reset(self):
        obs = self.env.reset()
        # print(obs.shape)
        # obs = self.gray_scale(obs)#/255.
        self.queue.append(obs)
        return np.stack(self.queue, axis=0)
    # def gray_scale(self, image):
    #     gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    #     return gray
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print(obs.shape)
        # cv2.imwrite("./obs.png", obs)
        # cv2.imshow("as s", obs)
        # cv2.waitKey()
        # obs = self.gray_scale(obs)#/255.
        obs = obs[36:,:]
        obs = cv2.resize(obs, (84,84), None)
        # cv2.imshow("obs", obs)
        # cv2.waitKey()
        # exit()
        reward = reward*2 + 0.001
        self.queue.append(obs)
        obs = np.stack(self.queue, axis=0)
        return obs, reward, done, info
    def render(self):
        return self.env.render()
    def close(self):
        self.env.close()