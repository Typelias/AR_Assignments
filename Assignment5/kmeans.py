from typing import Tuple
from typing import List
import matplotlib.pyplot as plt
from numpy import sqrt
import numpy as np

def centerToLists(center1, center2):
    return ([center1[0], center2[0]], [center1[1], center2[0]])


def drawPlot(center1, center2, gc1, gc2):
    plt.scatter([i[0] for i in gc1], [i[1] for i in gc1], c='blue')
    plt.scatter([i[0] for i in gc2], [i[1] for i in gc2], c='red')
    plt.scatter(center1[0], center1[1],c='lightblue')
    plt.scatter(center2[0], center2[1],c='orange')

def calcDistance(p1, p2):
    return sqrt((p1[0] -p2[0])**2 + (p1[1] - p2[1])**2)

def categorizePoints(points, center1, center2) -> Tuple[List, List]:
    gc1 = []
    gc2 = []
    for point in points:
        d1 = calcDistance(point, center1)
        d2 = calcDistance(point, center2)
        if d1 < d2:
            gc1.append(point)
        else:
            gc2.append(point)
    return (gc1, gc2)

def calculateNewCenter(gc1: List, gc2: List) -> Tuple[Tuple, Tuple]:
    g1x = np.asarray([i[0] for i in gc1]).sum()/len(gc1)
    g1y = np.asarray([i[1] for i in gc1]).sum()/len(gc1)

    g2x = np.asarray([i[0] for i in gc2]).sum()/len(gc2)
    g2y = np.asarray([i[1] for i in gc2]).sum()/len(gc2)

    return ((g1x, g1y), (g2x, g2y))
    


points = [
    (0, 0.5),
    (0, 0.75),
    (1, 1),
    (1.1, 0.4),
    (1, 5, 0.75),
    (2.5, 1),
    (3, 2),
    (4, 1.5),
    (4, 2.5),
    (5, 2)
]

center1 = (1,1.5)
center2 = (3,1)

(gc1, gc2) = categorizePoints(points, center1, center2)
drawPlot(center1, center2, gc1, gc2)
plt.show()
(center1, center2) = calculateNewCenter(gc1, gc2)
(gc1, gc2) = categorizePoints(points, center1, center2)
drawPlot(center1, center2, gc1, gc2)
plt.show()
(center1, center2) = calculateNewCenter(gc1, gc2)
(gc1, gc2) = categorizePoints(points, center1, center2)
drawPlot(center1, center2, gc1, gc2)
plt.show()




