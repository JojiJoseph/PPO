"""

Utility script to train diff gym compatible libraries with ppo

"""

from ppo import PPO

algo = PPO(learning_rate=1e-3, n_timesteps= 1e6, n_epochs=10, batch_size=256, n_rollout_timesteps=512, device="cpu", coeff_entropy=0.01)

algo.learn()

