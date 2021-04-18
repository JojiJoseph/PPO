import numpy as np
import tensorflow as tf

import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')


# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym

class Actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(64, activation="relu")
        self.l2 = tf.keras.layers.Dense(2)
    def call(self, x):
        y = self.l1(x)
        return self.l2(y)

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(64, activation="relu")
        self.l2 = tf.keras.layers.Dense(1)
    def call(self, x):
        y = self.l1(x)
        return self.l2(y)

actor = Actor()
critic = Critic()
optim = tf.keras.optimizers.Adam()
optim_critic = tf.keras.optimizers.Adam()
huber_loss = tf.keras.losses.Huber()
env = gym.make("CartPole-v0")

episodic_rewards = []
smoothened_rewards = []
gamma = 0.99
for episode in range(2000):
    done = False
    state = env.reset()

    states = []
    actions = []
    rewards = []

    episodic_reward = 0
    while not done:
        state = tf.expand_dims(state, 0)
        prob_distrib = actor(state)

        action = tf.random.categorical(prob_distrib, num_samples=1)[0,0].numpy().item()

        next_state, reward, done, _ = env.step(action)

        states.append(state[0])
        actions.append(action)
        rewards.append(reward)

        episodic_reward += reward

        state = next_state
    G = 0
    for i in range(len(rewards)-1,-1,-1):
        rewards[i] = rewards[i] + gamma*G
        G = rewards[i]
    episodic_rewards.append(episodic_reward)
    smoothened_rewards.append(sum(episodic_rewards[-10:])/len(episodic_rewards[-10:]))
    # print(states)
    # print(actions)
    # print(rewards)
    states = np.array(states)
    values = critic(states)
    advantages = rewards - tf.squeeze(values)
    mask = np.eye(2)[actions]
    init_probs = tf.nn.softmax(actor(states), axis=-1) + 1e-7
    init_probs = tf.reduce_sum(init_probs*mask,-1)
    for epoch in range(8):
        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(actor(states), axis=-1)
            # print(probs)
            # print(probs.shape)
            # print(mask.shape)
            probs = tf.reduce_sum(probs*mask, axis=-1)
            # print(probs
            loss = -tf.minimum((probs/init_probs)*advantages, tf.clip_by_value(probs/init_probs,0.8,1.2)*advantages)
            loss = tf.reduce_mean(loss)
            # print(loss)
            grads = tape.gradient(loss,actor.trainable_variables)
            optim.apply_gradients(zip(grads, actor.trainable_variables))
    with tf.GradientTape() as tape:
        values = critic(states)
        loss = huber_loss(values, tf.expand_dims(rewards, 0))
        grads = tape.gradient(loss, critic.trainable_variables)
        optim_critic.apply_gradients(zip(grads, critic.trainable_variables))
        # print(loss)
    print(episode, episodic_reward)



state = env.reset()
done = False
while not done:
    state = tf.expand_dims(state, 0)
    prob_distrib = actor(state)
    action = tf.random.categorical(prob_distrib, num_samples=1)[0,0].numpy().item()
    env.render()
    state, reward, done, _ = env.step(action)

plt.plot(episodic_rewards)
plt.savefig('ppo.png')
plt.close()
plt.plot(smoothened_rewards)
plt.savefig('ppo_avg.png')
# plt.show()