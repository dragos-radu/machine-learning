import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-5, 5, 100)
y=np.linspace(-5, 5, 100)

x, y = np.meshgrid(x, y)

z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(x, y, z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3d")

fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

plt.show()
