# TODO: Implement observation normalization and reward normalization

from ppo import PPO

algo = PPO(env_name="Pendulum-v0", learning_rate=1e-4, n_timesteps= 1e6, n_epochs=16, batch_size=256, n_rollout_timesteps=1024, device="cpu", coeff_entropy=0.0,
obs_normalization="simple",obs_scale=[1,1,8],rew_normalization="simple",rew_scale=8.1,rew_shift=8.1, action_scale=2)

algo.learn()