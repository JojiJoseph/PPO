import gym
import torch
import time

n_episodes = 2

fps = 30
from net import ActorCritic

env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor_critic = ActorCritic(state_dim, n_actions)
actor_critic.load_state_dict(torch.load("./LunarLander-v2.pt"))

for episode in range(5):
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