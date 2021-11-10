from matplotlib import image
import numpy as np
from matplotlib.image import imread
from matplotlib.image import imsave


def calculatePixelValue(imgArr, kernal):
    red = np.array([
        [imgArr[0][0][0]*kernal[0][0], imgArr[0][1][0] *
            kernal[0][1], imgArr[0][2][0]*kernal[0][2]],
        [imgArr[1][0][0]*kernal[1][0], imgArr[1][1][0] *
            kernal[1][1], imgArr[2][2][0]*kernal[1][2]],
        [imgArr[2][0][0]*kernal[2][0], imgArr[2][1][0]
            * kernal[2][1], imgArr[2][2][0]*kernal[2][2]]
    ]).sum()
    green = np.array([
        [imgArr[0][0][1]*kernal[0][0], imgArr[0][1][1] *
            kernal[0][1], imgArr[0][2][1]*kernal[0][2]],
        [imgArr[1][0][1]*kernal[1][0], imgArr[1][1][1] *
            kernal[1][1], imgArr[2][2][1]*kernal[1][2]],
        [imgArr[2][0][1]*kernal[2][0], imgArr[2][1][1]
            * kernal[2][1], imgArr[2][2][1]*kernal[2][2]]
    ]).sum()
    blue = np.array([
        [imgArr[0][0][0]*kernal[0][2], imgArr[0][1][2] *
            kernal[0][1], imgArr[0][2][2]*kernal[0][2]],
        [imgArr[1][0][0]*kernal[1][2], imgArr[1][1][2] *
            kernal[1][1], imgArr[2][2][2]*kernal[1][2]],
        [imgArr[2][0][0]*kernal[2][2], imgArr[2][1][2]
            * kernal[2][1], imgArr[2][2][2]*kernal[2][2]]
    ]).sum()
    red = red/255
    green = green/255
    blue = blue/255

    if(imgArr[1][1][3] >= 1):
        print(imgArr[1][1][3], "HEJ")

    return np.array([red, green, blue, imgArr[1][1][3]])


def apply(img, kernal):
    x_size, y_size, _ = img.shape
    output = np.zeros((x_size, y_size, 4))

    for y, val in enumerate(img):
        for x, val in enumerate(val):
            imgArr = np.zeros((3, 3, 4))
            try:
                imgArr[0][0] = img[y-1][x-1]
            except:
                imgArr[0][0] = [0., 0., 0., 0.]
            try:
                imgArr[0][1] = img[y-1][x]
            except:
                imgArr[0][1] = [0., 0., 0., 0.]
            try:
                imgArr[0][2] = img[y-1][x+1]
            except:
                imgArr[0][2] = [0., 0., 0., 0.]
            try:
                imgArr[1][0] = img[y][x-1]
            except:
                imgArr[1][0] = [0., 0., 0., 0.]
            try:
                imgArr[1][1] = img[y][x]
            except:
                imgArr[1][1] = [0., 0., 0., 0.]
            try:
                imgArr[1][2] = img[y][x+1]
            except:
                imgArr[1][2] = [0., 0., 0., 0.]
            try:
                imgArr[2][0] = img[y+1][x-1]
            except:
                imgArr[2][0] = [0., 0., 0., 0.]
            try:
                imgArr[2][1] = img[y+1][x]
            except:
                imgArr[2][1] = [0., 0., 0., 0.]
            try:
                imgArr[2][2] = img[y+1][x+1]
            except:
                imgArr[2][2] = [0., 0., 0., 0.]
            # print(imgArr[1][1])
            res = calculatePixelValue(imgArr, kernal)
            output[y][x] = res
            # print(np.amax(res))
            """ if y == 0:
                if x == 0:
                    imgArr = np.zeros((3, 3, 4))
                    imgArr[1][1] = img[0][0]
                    imgArr[1][2] = img[0][1]
                    imgArr[2][1] = img[1][0]
                    imgArr[2][2] = img[1][1]
                    img[y][x] = calculatePixelValue(imgArr, kernal)
                elif x == x_size-1:
                    imgArr = np.zeros((3, 3, 4))
                    imgArr[1][0] = img[0][x-1]
                    imgArr[1][1] = img[0][x]
                    imgArr[2][0] = img[1][x-1]
                    imgArr[2][1] = img[1][x]
                    img[y][x] = calculatePixelValue(imgArr, kernal)
                else:
                    imgArr = np.zeros((3, 3, 4))
                    imgArr[1][0] = img[0][x-1]
                    imgArr[1][1] = img[0][x]
                    imgArr[2][0] = img[1][x-1]
                    imgArr[2][1] = img[1][x]
                    img[y][x] = calculatePixelValue(imgArr, kernal) """

    return output


img = imread("ill.png")
sharp = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

s = apply(img, sharp)

print(np.max(s))


imsave("sharp.png", s)
