"""

Utility script to train diff gym compatible libraries with ppo

"""

from ppo import PPO

algo = PPO(learning_rate=1e-3, n_timesteps= 1e6, n_epochs=16, batch_size=128, n_rollout_timesteps=256)

algo.learn()

