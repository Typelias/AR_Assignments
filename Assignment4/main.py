import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("right.jpg")
imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("left.jpg")
imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgGray1, None)
kp2, des2 = sift.detectAndCompute(imgGray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

matches = np.asarray(matches)

if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError("Not enough points")

print(H)
print(dst[0])
print(src[0])
dst = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
plt.subplot(122), plt.imshow(dst), plt.title("Warped Image")
plt.show()
plt.figure()
dst[0:img2.shape[0], 0:img2.shape[1]] = img2
cv2.imwrite("output.jpg", dst)
plt.imshow(dst)
plt.show()

images = [img1, img2]
stitcher = cv2.Stitcher_create()

status, result = stitcher.stitch(images)

plt.figure(figsize=[30,10])
plt.imshow(result)
cv2.imwrite("cv2Stitch.jpg", result)

