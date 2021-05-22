import gym
import torch
import time

n_episodes = 2

fps = 30
from net import ActorCritic

env = gym.make("CartPole-v1")

actor_critic = ActorCritic(4, 2)
actor_critic.load_state_dict(torch.load("./CartPole-v1.pt"))

state = env.reset()
done = False
while not done:
    state = state[None,:]
    state = torch.tensor(state).float()#.cuda()

    action_params, _ = actor_critic(state)
    action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
    action = action[0].detach().cpu().numpy()
    env.render()
    time.sleep(1/fps)
    next_state, reward, done, info = env.step(action)
    state = next_state
env.close()