import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

N = 256
angle = np.linspace(0, 8 * 2 * np.pi, N)
radius = np.linspace(.5, 1., N)
X = radius * np.cos(angle)
Y = radius * np.sin(angle)
plt.scatter(X, Y, c = angle, cmap = cm.Pastel1)
plt.axis('off')
plt.savefig("Pastel.svg")
plt.show()
