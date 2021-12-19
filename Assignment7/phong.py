import numpy as np
import matplotlib.pyplot as plt

"""
Formula:
I = kA*IA + kD*ID + kS*(cos(𝜑)^𝛼*IL)
"""


def phong(kA, IA, kD, ID, kS, phi, alpha, IL):
    return (kA*IA) + (kD+ID) + (kS * (np.cos(phi)**alpha)*IL)


IA = 0.3 # Ambient in the room
ID = 0.2 # How much light we add on the whole pov
IL = 1.2 # Intesity 

kA = 0.1 # How much ambient light it reflects
kD = 0.1 # How the defuse lighting it reflects
kS = 1.0 # 

alpha = np.linspace(0.0, 20, num=20)
phi = np.linspace(0.0, np.pi/2, num=20)

ya = [phong(kA, IA, kD, ID, kS, np.pi/4, x, IL) for x in alpha]
yp = [phong(kA, IA, kD, ID, kS, x, 3, IL) for x in phi]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(alpha, ya, 'b')
ax2.plot(phi, yp, 'g')
plt.show()
