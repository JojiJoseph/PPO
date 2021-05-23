"""

Utility script to train diff gym compatible libraries with ppo

"""

from ppo import PPO

algo = PPO(learning_rate=1e-3, n_timesteps= 1e6, n_epochs=4, batch_size=64, n_rollout_timesteps=1024, device="cpu", env_name="MountainCarContinuous-v0",
obs_normalization="simple", obs_shift=[0.5,0], obs_scale=[0.5,0.04])

algo.learn()

