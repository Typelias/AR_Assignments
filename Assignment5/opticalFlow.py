from math import sqrt
import numpy as np
import gc
from PIL import Image
 
def I(pixel, opt=0):
    R, G, B = pixel
    return (0.2126 * R) + (0.7152 * G) + (0.0722 * B)
 
 
def image_to_intensity(image, opt=0):
    pix = image.load()
    return np.asarray([[I(pix[y, x]) for y in range(image.size[0])] for x in range(image.size[1])])
 
 
def lucas_kanade(im1, im2, win=2):
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow
 

 
n1 = "img1.jpg"
n2 = "img2.jpg"

im1 = Image.open(n1)
im2 = Image.open(n2)

i1 = image_to_intensity(im1)
i2 = image_to_intensity(im2)

win = 5

opfl = lucas_kanade(i1, i2, win)

# check GC, suprisingly values returned after each iteration are exactly the same, or less than 1% different
print (len(gc.get_objects()))
gc.collect()
print (len(gc.get_objects()))

# prepare iblack-red image, presenting the optical flow and blend it with original one
normal = 255.0 / ((2*win+1)*2**(0.5))

width, height = im1.size

out = Image.new("RGB", (width,height))
pix = out.load()
for y in range(height):
    for x in range(width):
        pix[x,y] = (min(255,int(normal*sqrt(abs(opfl[y][x][0])*abs(opfl[y][x][0]) + abs(opfl[y][x][1])*abs(opfl[y][x][1])))),0,0)

out.save('a'+n2)
i3 = Image.blend(im2, out, 0.2)
i3 = i3.point(lambda i: i * 2)
i3.save('b'+n2)