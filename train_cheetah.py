"""

Utility script to train diff gym compatible libraries with ppo

"""
import pybullet_envs

from ppo import PPO

algo = PPO(env_name="HalfCheetahBulletEnv-v0", learning_rate=3e-5, n_timesteps= 2e6, n_epochs=20, batch_size=256, n_rollout_timesteps=1024, device="cpu",clip_range=0.2, max_grad_norm=0.5)

algo.learn()

