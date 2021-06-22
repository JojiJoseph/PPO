from typing import Deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading results
df = pd.read_csv("./results/ch_welford/result.csv")
x1 = df.values[:,1]
y1 = df.values[:,2]
df = pd.read_csv("./results/ant_welford/result.csv")
x2 = df.values[:,1]
y2 = df.values[:,2]
df = pd.read_csv("./results/hopper_welford/result.csv")
x3 = df.values[:,1]
y3 = df.values[:,2]
df = pd.read_csv("./results/walker_welford/result.csv")
x4 = df.values[:,1]
y4 = df.values[:,2]

def smooth(y, q_size=100):
    q = Deque(maxlen=q_size)
    y_avg = []
    for item in y:
        q.append(item)
        y_avg.append( np.mean(q) )
    return y_avg

y1_smooth = smooth(y1)
y2_smooth = smooth(y2)
y3_smooth = smooth(y3)
y4_smooth = smooth(y4)

# Plotting Results

# Without smoothening the curves
plt.subplot(1,2,1)
plt.plot(x1, y1, label="HalfCheetah")
plt.title("HalfCheetah")
plt.subplot(1,2,2)
plt.plot(x2, y2, label="Ant")
plt.title("Ant")
plt.show()

plt.subplot(1,2,1)
plt.plot(x3, y3, label="Hopper")
plt.title("Hopper")
plt.subplot(1,2,2)
plt.plot(x4, y4, label="Walker2D")
plt.title("Walker2D")
plt.show()

# After smoothening the curves
plt.subplot(1,2,1)
plt.plot(x1, y1_smooth, label="HalfCheetah")
plt.title("HalfCheetah")
plt.ylabel("Score")
plt.xlabel("Timesteps")
plt.grid()
plt.subplot(1,2,2)
plt.plot(x2, y2_smooth, label="Ant")
plt.title("Ant")
plt.ylabel("Score")
plt.xlabel("Timesteps")
plt.grid()
plt.savefig("cheetah_ant.pdf", orientation="landscape")
plt.show()

plt.subplot(1,2,1)
plt.plot(x3, y3_smooth, label="Hopper")
plt.title("Hopper")
plt.ylabel("Score")
plt.xlabel("Timesteps")
plt.grid()
plt.subplot(1,2,2)
plt.plot(x4, y4_smooth, label="Walker2D")
plt.title("Walker2D")
plt.ylabel("Score")
plt.xlabel("Timesteps")
plt.grid()
plt.show()