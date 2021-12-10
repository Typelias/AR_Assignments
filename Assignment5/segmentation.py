import numpy as np
import cv2
from typing import List, Tuple
import collections
import random
import math
from PIL import Image

class Point:
    position = (0,0)
    color = [0,0,0]
    category = [0,0,0]


def convImForShow(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def calcDistance(p1, p2) -> float:
    x = np.square(p1[0] - p2[0])
    y = np.square(p1[1] - p2[1])
    z = np.square(p1[2] - p2[2])
    sum = x+y+z
    return math.sqrt(sum)

def categorizePoints(points: List[Point], centers: List) -> Tuple[List[List], List[Point]]:
    reee = set()
    ret = [[] for i in range(len(centers))]
    dist = np.zeros(len(centers))
    l = []
    for p in points:
        for i, c in enumerate(centers):
            dist[i] = calcDistance(p.color, c)
        index = np.argmin(dist)
        reee.add(index)
        ret[index].append(p.color)
        p.category = centers[index]
        # print("----------------------------------------")
        # print(p.color)
        # print(p.position)
        # print(p.category)
        # print("----------------------------------------")
        # print()
        l.append(p)
    #print(reee)
    return (ret, l)

def calcCenter(category: List) -> List:
    x = np.asarray([i[0] for i in category]).sum()/len(category)
    y = np.asarray([i[1] for i in category]).sum()/len(category)
    z = np.asarray([i[2] for i in category]).sum()/len(category)
    return [x,y,z]


def calcNewCenters(categories: List[List]) -> List[List]:
    newCenters = [[] for i in range(len(categories))]
    for i, cat in enumerate(categories):
        newCenters[i] = calcCenter(cat)
    return newCenters

def colorPoints(points: List[List], categories: List[List], centers: List[List]) -> List[List]:
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    for i, cat in enumerate(categories):
        for p in cat:
            indexes = [i for i,x in enumerate(points) if compare(x,p)]
            for j in indexes:
                points[j] = centers[i]
    return points

def imageToPoints(img) -> List[Point]:
    ret = []
    for y in range(len(img)):
        for x in range(len(img[0])):
            temp = Point()
            temp.position = (x, y)
            temp.color = img[y][x]
            ret.append(temp)
    return ret

def pointsToImage(points: List[Point], shape):
    p = [x.category for x in points]
    return np.asarray(p, np.uint8).reshape(shape)

def compareCenters(oldCenter, newCenter):
    underLimit = 0
    for i, _ in enumerate(oldCenter):
        distance = calcDistance(oldCenter[i], newCenter[i])
        if distance < 1:
            underLimit += 1
    
    if underLimit > (len(oldCenter)*0.7):
        return True
    return False


im = Image.open("im1.jpg")
im = np.asarray(im)
numberOfClusters = 8

numberOfIterations = 50

shape = im.shape
points = imageToPoints(im)
randPoints = random.sample(points, numberOfClusters)
centers = [p.color for p in randPoints]

cat, points = categorizePoints(points, centers)

iters = 0
for i in range(numberOfIterations):
    cat, points = categorizePoints(points, centers)
    newCenters = calcNewCenters(cat)
    iters = i
    if compareCenters(centers, newCenters):
        break
    centers = newCenters

print("Did ", iters, " iterations")

temp = pointsToImage(points, shape)
temp = Image.fromarray(temp)
temp.show("Iter 0")
temp.save("meme.png")


# im = cv2.imread("im1.jpg")

# shape = im.shape
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# numberOfClusters = 5

# numberOfIterations = 10

# points = imageToPoints(im)
# # img = pointsToImage(points, shape)




# randPoints = random.sample(points, numberOfClusters)
# shape = im.shape
# centers = [p.category for p in randPoints]

# cat, points = categorizePoints(points, centers)
# temp = pointsToImage(points, shape)
# print(points[0].category)
# print(points[0].color)
# print(temp[0,0])
# cv2.imshow("meme1", convImForShow(temp))
# centers = np.asarray(calcNewCenters(cat), int)
# print("------------------------")
# cat, points = categorizePoints(points, centers)
# temp = pointsToImage(points, shape)
# print(points[0].category)
# print(points[0].color)

# print(temp[0,0])

# #cv2.imshow("meme", convImForShow(temp))

# cv2.waitKey()






