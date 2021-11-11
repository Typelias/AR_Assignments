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
    kH, kW = kernel.shape[:2]
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

sharp = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

blurr = np.array(([
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]]),dtype=float)

edge = np.array(([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]),dtype="int")


img = cv2.imread("ill.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

save = Image.fromarray(img)
save.save('original.png')


sharpend = convolve(img, sharp)
blurred = convolve(img, blurr)
edged = convolve(img, edge)

save = Image.fromarray(sharpend)
save.save('sharp.png')

save = Image.fromarray(blurred)
save.save('blurr.png')

save = Image.fromarray(edged)
save.save('edge.png')



