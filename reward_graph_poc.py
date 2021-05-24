import numpy as np
import matplotlib.pyplot as plt


y = np.random.randint(0,100, (100,))
y_max = y + np.random.randint(0,50, (100,))
y_min = y - np.random.randint(0,50, (100,))

plt.plot(y)
plt.fill_between(range(100),y_min, y_max)
plt.show()