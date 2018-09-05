import numpy as np
import random

x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

print("x_data:", x_data)
print("len(x_data[0])", len(x_data[0]))
