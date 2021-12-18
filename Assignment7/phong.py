import numpy as np
import matplotlib.pyplot as plt

"""
Formula:
I = kA*IA + kD*ID + kS*(cos(𝜑)^𝛼*IL)
"""


def phong(kA, IA, kD, ID, kS, phi, alpha, IL):
    return (kA*IA) + (kD+ID) + (kS * (np.cos(phi)**alpha)*IL)


IA = 6.4
ID = 3.9
IL = 2.5

kA = 5.
kD = 2.
kS = 4.

alpha = np.linspace(0.0, 3.0, num=20)
phi = np.linspace(0.0, np.pi/2, num=20)

ya = [phong(kA, IA, kD, ID, kS, np.pi/2, x, IL) for x in alpha]
yp = [phong(kA, IA, kD, ID, kS, x, 3, IL) for x in phi]

plt.plot(alpha, ya, 'b')
plt.plot(phi, yp, 'g')
plt.show()
