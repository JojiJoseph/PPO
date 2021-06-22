
import gym
import numpy as np
from typing import Deque

class FrameStackWrapper:
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.queue = Deque(maxlen=4)
        self.queue.append(np.zeros([96,96], np.float32))
        self.queue.append(np.zeros([96,96], np.float32))
        self.queue.append(np.zeros([96,96], np.float32))
        self.queue.append(np.zeros([96,96], np.float32))
    def reset(self):
        obs = self.env.reset()
        obs = self.gray_scale(obs)/255.
        self.queue.append(obs)
        return np.stack(self.queue, axis=0)
    def gray_scale(self, image):
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        return gray
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.gray_scale(obs)/255.
        self.queue.append(obs)
        obs = np.stack(self.queue, axis=0)
        return obs, reward, done, info
    def render(self):
        return self.env.render()
    def close(self):
        self.env.close()


if __name__ == "__main__":
    from net import CnnActorCriticContinuos
    import torch
    env = gym.make("CarRacing-v0")
    actor_critic = CnnActorCriticContinuos(4,env.action_space.shape[0])
    env = FrameStackWrapper(env)
    state = env.reset()
    state = state[None,:]
    print(state.shape)
    state = torch.from_numpy(state).float()
    print(state)
    out = actor_critic(state)
    print(out)