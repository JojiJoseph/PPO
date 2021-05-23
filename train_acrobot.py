# TODO: Implement observation normalization and reward normalization

from ppo import PPO

algo = PPO(env_name="Acrobot-v1", learning_rate=1e-4, n_timesteps= 1e6, n_epochs=10, batch_size=256, n_rollout_timesteps=1024, device="cpu", coeff_entropy=0.01)

algo.learn()