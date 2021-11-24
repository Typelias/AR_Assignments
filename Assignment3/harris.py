import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris(img_name, window_size, k, threshold):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    h = img.shape[0]
    w = img.shape[1]

    matrix_r = np.zeros((h, w))

    dx = cv2.Sobel(img_gauss, cv2.CV_64F,1,0, ksize=3)
    dy = cv2.Sobel(img_gauss, cv2.CV_64F, 0, 1, ksize=3)

    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    offset = int(window_size/2)
    print("Finding corners...")
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
            sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])

            H = np.array([[sx2, sxy], [sxy, sy2]])
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_r[y - offset, x - offset] = R

    cv2.normalize(matrix_r, matrix_r, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            value = matrix_r[y, x]
            if value > threshold:
                cv2.circle(img, (x, y), 3, (0, 255, 0))

    plt.figure("Harris detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("My Harris")
    plt.xticks([]), plt.yticks([])
    plt.show()

def cv2Harris(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    img[harris > 0.01 * harris.max()] = [0, 0, 255]

    plt.figure("Harris detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("CV2 Harris")
    plt.xticks([]), plt.yticks([])
    plt.show()

def cv2Sift(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv2.drawKeypoints(img, kp, img)
    plt.figure("SIFT Detector")
    plt.imshow(img), plt.title("SIFT")
    plt.xticks([]), plt.yticks([])
    plt.show()

img_name = "start2.jpeg"

harris(img_name, 5, 0.04, 0.30)
cv2Harris(img_name)
cv2Sift(img_name)
