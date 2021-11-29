import cv2
import numpy as np

def depth(img1, img2, distance):
    height, width = img1.shape
    disparity = 0
    depthArray = np.zeros((height, width))
    maxDepth = 255/distance
    for y in range(0,height):
        for x in range(0, width):
            prev_min_val = np.inf
            for d in range(distance):
                temp_min_value = float(img1[y][x]) - float(img2[y][x-d])
                min_value = temp_min_value * temp_min_value
                if min_value < prev_min_val:
                    prev_min_val = min_value
                    disparity = d
            depthArray[y][x] = disparity * maxDepth
    cv2.imwrite("sd.png", depthArray)






img1 = cv2.imread("ps1.ppm", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("ps2.ppm", cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=5)
disp = stereo.compute(img1,img2)
cv2.imwrite("cv2Sereo.png", disp)

depth(img1, img2, 16)
