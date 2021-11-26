import numpy as np
np.set_printoptions(suppress=True)

def calcHomo(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

def calcNewPoints(src, H):
    newPoints = []
    for x,y in src:
        vec = np.asarray([x,y,1])
        mult = H.dot(vec)
        newPoint = mult/mult[-1]
        newPoints.append([newPoint[0], newPoint[1]])

        
    return np.asarray(newPoints)

src = np.asarray([[0,0],[0,3],[5,3],[5,0]])
dst = np.asarray([[1,1],[3,3],[6,3],[5,2]])

H = calcHomo(src, dst)

print(H)

print(calcNewPoints(src, H))

