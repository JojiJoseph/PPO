import gym
import torch
import time

from gym.wrappers import Monitor

n_episodes = 2

fps = 30
from net import ActorCritic

# env = gym.make("MountainCar-v0")
env = Monitor(gym.make('MountainCar-v0'), './video', force=True)

actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n)
actor_critic.load_state_dict(torch.load("./MountainCar-v0.pt"))

state = env.reset()
done = False
total_reward = 0
while not done:
    state[0] += 0.5
    state[0] /= 0.5
    # state[1] += 0.04
    state[1] /= 0.04
    state = state[None,:]
    state = torch.tensor(state).float()#.cuda()

    action_params, _ = actor_critic(state)
    action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
    action = action[0].detach().cpu().numpy()
    env.render()
    time.sleep(1/fps)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
env.close()
print(total_reward)