"""

Utility script to train diff gym compatible libraries with ppo

"""

from ppo_poc_mountaincar import PPO

algo = PPO(learning_rate=1e-3, n_timesteps= 1e6, n_epochs=4, batch_size=64, n_rollout_timesteps=1024, device="cpu", env_name="MountainCar-v0")

algo.learn()

