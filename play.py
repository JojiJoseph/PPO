from gym.wrappers.atari_preprocessing import AtariPreprocessing
from frame_stack_atari import AtariFrameStackWrapper
from atari_wrapper import AtariRamWrapper, BreakoutBlindWrapper
from frame_stack_wrapper import FrameStackWrapper
import torch
import gym
from gym.wrappers import Monitor
import argparse
import time
import yaml
import pybullet_envs
import numpy as np

from net import ActorCritic, ActorCriticContinuous, CnnActorCriticContinuos, CnnAtari

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp",type=str, required=True,help="The experiment name as defined in the yaml file")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()
experiment = args.exp
hyperparams = experiments[experiment]
fps = 30

env = Monitor(gym.make(hyperparams['env_name']), './video', force=True)

print(env.action_space)
print(env.observation_space)

if "wrappers" in hyperparams:
    if "frame_stack" in hyperparams["wrappers"]:
        env = FrameStackWrapper(env)
    if "atari_ram_wrapper" in hyperparams["wrappers"]:
        env = AtariRamWrapper(env)
    if "atari_wrapper" in hyperparams["wrappers"]:
        env = AtariFrameStackWrapper(AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, terminal_on_life_loss=False, scale_obs=True))
    if "breakout_blind_wrapper" in hyperparams["wrappers"]:
        env = BreakoutBlindWrapper(AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, terminal_on_life_loss=False, scale_obs=True))
state_dim = env.observation_space.shape[0]
size = 64
if "net_size" in hyperparams:
    size = hyperparams["net_size"]
if "action_scale" in hyperparams:
    action_scale = hyperparams["action_scale"]
else:
    action_scale = 1
if type(env.action_space) == gym.spaces.Discrete:
    n_actions = env.action_space.n
    actor_critic = ActorCritic(state_dim, n_actions, size=size)
    if "policy" in hyperparams and hyperparams["policy"] == "cnn_atari":
        actor_critic = CnnAtari(n_actions)
elif type(env.action_space) == gym.spaces.Box:
    action_dim = env.action_space.shape[0]
    actor_critic = ActorCriticContinuous(state_dim, action_dim, action_scale=action_scale, size=size)
if "policy" in hyperparams and hyperparams["policy"] == "cnn_car_racing":
    actor_critic = CnnActorCriticContinuos(4, action_dim)
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
if "obs_normalization" in hyperparams:
    if obs_normalization == "welford":
        welford_mean = actor_critic.welford_mean.data.detach().numpy()
        welford_M2 = actor_critic.welford_M2.data.detach().numpy()
        welford_count = actor_critic.welford_count.data.detach().numpy()

def normalize_obs(observation):
    if obs_normalization == "simple":
        if obs_shift is not None:
            observation += obs_shift
        if obs_scale is not None:
            observation /= obs_scale
    if obs_normalization == "welford":
        observation = (observation - welford_mean)/np.sqrt(welford_M2/welford_count)
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