import torch
import gym
from gym.wrappers import Monitor
import argparse
import time
import yaml
import pybullet_envs

from net import ActorCritic, ActorCriticContinuous

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp",type=str, required=True,help="The experiment name as defined in the yaml file")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()
experiment = args.exp
hyperparams = experiments[experiment]
fps = 30

env = Monitor(gym.make(hyperparams['env_name']), './video', force=True)

state_dim = env.observation_space.shape[0]

if type(env.action_space) == gym.spaces.Discrete:
    n_actions = env.action_space.n
    actor_critic = ActorCritic(state_dim, n_actions)
elif type(env.action_space) == gym.spaces.Box:
    action_dim = env.action_space.shape[0]
    actor_critic = ActorCriticContinuous(state_dim, action_dim)

actor_critic.load_state_dict(torch.load("./results/" + experiment + "/model.pt"))

obs_normalization = None
obs_shift = 0
obs_scale = 0
if "obs_normalization" in hyperparams:
    obs_normalization = hyperparams["obs_normalization"]
if "obs_shift" in hyperparams:
    obs_shift = hyperparams["obs_shift"]
if "obs_scale" in hyperparams:
    obs_scale = hyperparams["obs_scale"]

def normalize_obs(observation):
    if obs_normalization == "simple":
        if obs_shift is not None:
            observation += obs_shift
        if obs_scale is not None:
            observation /= obs_scale
    return observation

try:
    env.render() # Should call render function before reset for pybullet environments
except:
    pass

state = env.reset()
done = False
total_reward = 0
while not done:
    state = normalize_obs(state)
    state = state[None,:]
    state = torch.tensor(state).float()#.cuda()

    action_params, _ = actor_critic(state)
    if type(env.action_space) == gym.spaces.Discrete:
        action = torch.distributions.Categorical(logits=action_params[0]).sample((1,))
    else:
        mu, log_sigma = action_params
        distrib = torch.distributions.Normal(mu[0], log_sigma.exp())
        action = distrib.sample((1,))
    action = action[0].detach().cpu().numpy()
    env.render()
    time.sleep(1/fps)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
env.close()

print("Total Reward:", total_reward)