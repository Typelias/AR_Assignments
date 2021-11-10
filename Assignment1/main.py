import matplotlib.pyplot as plt
import numpy as np
import math

def calculateFOV(f, w):
    return math.degrees(2*math.atan(w/(2*f)))

def createPlot(ax, w):
    maxF = 100
    x = np.arange(1, maxF, 0.1)
    y = []
    for f in x:
        y.append(calculateFOV(f,w))
    ax.plot(x,y, label="Camera with w=" + str(w) + "mm")

w1 = 5
w2 = 40

fig, ax = plt.subplots()

createPlot(ax, w1)
createPlot(ax, w2)

plt.ylabel("Field of View (°)")
plt.xlabel("Focal lenght (mm)")
plt.legend()

plt.title("Field of view of camera as a function of the focal length")

exportPath = "Q2.png"

plt.savefig(exportPath)
