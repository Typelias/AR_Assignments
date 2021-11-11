from matplotlib import image
import numpy as np
from matplotlib.image import imread
from matplotlib.image import imsave
from PIL import Image
from numpy.core.fromnumeric import shape
import cv2
from skimage.exposure import rescale_intensity
from skimage.exposure.exposure import _output_dtype


def convolve(img, kernel):
    iH, iW = img.shape[:2]
    _, kW = kernel.shape[:2]
    pad = (kW - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))

    output = (output * 255).astype("uint8")
    return output


def convolveRGB(img, kernel):
    iH, iW = img.shape[:2]
    _, kW = kernel.shape[:2]
    pad = (kW - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW, 3), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            r = (roi[:, :, 0] * kernel).sum()
            g = (roi[:, :, 1] * kernel).sum()
            b = (roi[:, :, 2] * kernel).sum()
            k = [r, g, b]
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))

    output = (output * 255).astype("uint8")
    return output


roi = np.array(([
    [[250, 253, 251], [250, 253, 251], [250, 253, 251]],
    [[250, 253, 251], [250, 253, 251], [250, 253, 251]],
    [[250, 253, 251], [250, 253, 251], [250, 253, 251]]
]), dtype="int")


sharp = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

blurr = np.array(([
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]]), dtype=float)

edge = np.array(([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]), dtype="int")

rightSobel = np.array(([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]), dtype="int")

emboss = np.array(([
    [-2, 1, 0],
    [-1, 1, 1],
    [0, 1, 2]
]), dtype="int")

gblurr = np.array(([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4]
]), dtype="float")

gblurr = gblurr * (1/256)

umask = np.array(([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4]
]), dtype="float")

umask = umask * (-1/256)


""" img = cv2.imread("ill.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


shaprened = convolveRGB(img, sharp)
save = Image.fromarray(shaprened)
save.save('sharpened.png') """


img = cv2.imread("bilder/ill.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

save = Image.fromarray(img)
save.save('original.png')


print("Sharpening")
sharpend = convolveRGB(img, sharp)
save = Image.fromarray(sharpend)
save.save('sharp.png')

print("Blurring")
blurred = convolveRGB(img, blurr)
save = Image.fromarray(blurred)
save.save('blurr.png')

print("Detecting edges")
edged = convolveRGB(img, edge)
save = Image.fromarray(edged)
save.save('edge.png')

print("Right sobel")
rSobel = convolveRGB(img, rightSobel)
save = Image.fromarray(rSobel)
save.save('right_sobel.png')

print("Emboss")
em = convolveRGB(img, emboss)
save = Image.fromarray(em)
save.save('embossed.png')

print("G-blurr")
gb = convolveRGB(img, gblurr)
save = Image.fromarray(gb)
save.save('gblurr.png')

print("umask")
um = convolveRGB(img, umask)
save = Image.fromarray(um)
save.save('umask.png')
