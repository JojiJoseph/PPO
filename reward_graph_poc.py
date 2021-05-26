from typing import Deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./results/cartpole_1/result.csv")
x1 = df.values[:,1]
y1 = df.values[:,2]
df = pd.read_csv("./results/cartpole_1_e1/result.csv")
x2 = df.values[:,1]
y2 = df.values[:,2]
df = pd.read_csv("./results/cartpole_1_e2/result.csv")
x3 = df.values[:,1]
y3 = df.values[:,2]
df = pd.read_csv("./results/cartpole_1_e3/result.csv")
x4 = df.values[:,1]
y4 = df.values[:,2]
# df = pd.read_csv("cartpole_1_1.csv")
# x1 = df.values[:,1]
# y1 = df.values[:,2]
# avg_y = np.convolve(y, np.ones(10)/10, mode='same') # https://stackoverflow.com/a/22621523
# q = Deque(maxlen=10)
# y_mean = []
# y_max = []
# y_min = []
# for item in y:
#     q.append(item)
#     mean = np.mean(q)
#     y_mean.append(mean)
    # std = np.std(q)
    # print(std)
    # # y_max.append(mean + std)
    # # y_min.append(mean - std)
    # y_max.append(np.max(q))
    # y_min.append(np.min(q))

# avg_y = 0
# deq 
# y = np.random.randint(0,100, (100,))
# y_max = y + np.random.randint(0,50, (100,))
# y_min = y - np.random.randint(0,50, (100,))
def smooth(y, q_size=10):
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
# plt.plot(x,y)
plt.plot(x1, y1, label="exp1")
plt.plot(x2, y2, label="exp2")
# plt.plot(x3, y3_smooth, label="exp3")
# plt.plot(x4, y4, label="exp4")
plt.legend()
# plt.fill_between(x,y_min, y_max)
plt.show()
plt.plot(x1, y1_smooth, label="exp1")
plt.plot(x2, y2_smooth, label="exp2")
plt.plot(x3, y3_smooth, label="exp3")
plt.plot(x4, y4, label="exp4")
plt.legend()
# plt.fill_between(x,y_min, y_max)
plt.show()