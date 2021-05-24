import gym
import torch
import time

n_episodes = 2

fps = 30
from net import ActorCritic

env = gym.make("Acrobot-v1")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor_critic = ActorCritic(state_dim, n_actions)
actor_critic.load_state_dict(torch.load("./Acrobot-v1.pt"))
total_reward = 0
for episode in range(1):
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
        total_reward += reward
        state = next_state
env.close()
print(total_reward)