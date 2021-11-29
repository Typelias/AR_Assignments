import cv2
import numpy as np
import random
np.set_printoptions(suppress=True)

def calcHomo(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0][0], p1[i][0][1]
        u, v = p2[i][0][0], p2[i][0][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

def geoDistance(p1, p2, h):
    p1 = p1[0]
    p2 = p2[0]
    p1 = np.append(p1, 1)
    p2 = np.append(p2, 1)
    estimatep2 = np.dot(h, p1)
    estimatep2 = estimatep2/estimatep2[-1]
    error = p2 - estimatep2

    return np.linalg.norm(error)


def ransac(src, dst, threshold):
    maxInliers = []
    finalH = None
    meme = 0
    for i in range(1000):
        meme = i
        p1 = []
        p2 = []
        #First cord:
        index = random.randrange(0, len(src))
        p1.append(src[index])
        p2.append(dst[index])
        #Second cord:
        index = random.randrange(0, len(src))
        p1.append(src[index])
        p2.append(dst[index])
        #Third cord:
        index = random.randrange(0, len(src))
        p1.append(src[index])
        p2.append(dst[index])
        #Fourth cord:
        index = random.randrange(0, len(src))
        p1.append(src[index])
        p2.append(dst[index])

        h = calcHomo(p1, p2)
        inlliers = []

        for i in range(len(src)):
            d = geoDistance(src[i], dst[i], h)
            if d<5:
                inlliers.append([src[i], dst[i]])
        
        if len(inlliers) > len(maxInliers):
            maxInliers = inlliers
            finalH = h
        if len(maxInliers) > (len(src) * threshold):
            break
    print("Num iter:",meme)
    return finalH



img1 = cv2.imread("right.jpg")
img2 = cv2.imread("left.jpg")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create(nfeatures=2000)

keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)


src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


H2, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
H = ransac(src_pts, dst_pts, 0.74)

print(H)
print(H2)

result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

scale_percent = 40 # percent of original size
width = int(result.shape[1] * scale_percent / 100)
height = int(result.shape[0] * scale_percent / 100)
dim = (width, height)

result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Result", result)
cv2.imwrite("output.jpg", result)
cv2.waitKey()