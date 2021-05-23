import gym
import torch
import time

from gym.wrappers import Monitor

n_episodes = 2

fps = 30
from net import ActorCritic, ActorCriticContinuous

# env = gym.make("MountainCar-v0")
env = Monitor(gym.make('MountainCarContinuous-v0'), './video', force=True)

actor_critic = ActorCriticContinuous(env.observation_space.shape[0], env.action_space.shape[0])
actor_critic.load_state_dict(torch.load("./MountainCarContinuous-v0.pt"))

state = env.reset()
done = False
total_reward = 0
while not done:
    state[0] += 0.5
    state[0] /= 0.5
    # state[1] += 0.04
    state[1] /= 0.04
    state = state[None,:]
    # state = torch.tensor(state).float()#.cuda()

    with torch.no_grad():
        state = torch.as_tensor(state).float()#.to(device)

        # action_params, _ = actor_critic(state)
        # action = torch.argmax(torch.softmax(action_params[0],-1))
        prob_params, _ = actor_critic(state)
        # distrib = torch.distributions.Categorical(logits=prob_params[0])
        mu, log_sigma = prob_params
        # print(mu, log_sigma)
        distrib = torch.distributions.Normal(mu[0], log_sigma.exp())
        action = distrib.sample((1,))
        action = torch.clip(action, -1, 1)
        # print(action)
        log_prob = distrib.log_prob(action).sum(dim=1).item()

        action = action[0].cpu().numpy()
        # action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))[0]
        # action = action.cpu().numpy()
        # print(action)
    env.render()
    state, reward, done, info = env.step(action)
    total_reward += reward
    time.sleep(1/fps)
env.close()
print(total_reward)