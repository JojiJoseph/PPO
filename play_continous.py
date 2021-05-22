import gym
import torch
import time

n_episodes = 2

fps = 30
from net import ActorCriticContinous

env = gym.make("LunarLanderContinuous-v2")

print(env.action_space.shape)
print(env.observation_space.shape)

actor_critic = ActorCriticContinous(8, 2)
actor_critic.load_state_dict(torch.load("./LunarLanderContinuous-v2.pt"))

state = env.reset()
done = False
while not done:
    state = state[None,:]
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
        # print(action)
        log_prob = distrib.log_prob(action).sum(dim=1).item()

        action = action[0].cpu().numpy()
        # action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))[0]
        # action = action.cpu().numpy()
        # print(action)
    env.render()
    state, reward, done, info = env.step(action)
    time.sleep(1/fps)
env.close()