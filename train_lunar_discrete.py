"""

Utility script to train diff gym compatible libraries with ppo

"""

from ppo import PPO

algo = PPO(env_name="LunarLander-v2", learning_rate=1e-3, n_timesteps= 1e6, n_epochs=4, batch_size=256, n_rollout_timesteps=1024, device="cpu")

algo.learn()
