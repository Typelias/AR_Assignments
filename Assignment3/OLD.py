import numpy as np
from numpy import array
from PIL import Image
from matplotlib import pyplot as plt

def get_guassian_kernel(n, sigma=None):
    if sigma == None:
        sigma = 0.3 * (n // 2) + 0.8
    X = np.arange(-(n//2), n//2+1)
    kernel = np.exp(-(X**2)/(2*sigma**2))
    return kernel

def calc_derivative_conv(Image, filter_x, filter_y):
    h, w = Image.shape[:2]
    n = filter_x.shape[0]//2
    Image_a = np.zeros(Image.shape)
    Image_b = np.zeros(Image.shape)
    for x in range(n, w-n):
        patch = Image[:, x-n:x+n+1]
        Image_a[:, x] = np.sum(patch * filter_x, 1)
    filter_y = np.expand_dims(filter_y, 1)
    for y in range(n, h-n):
        patch = Image_a[y-n:y+n+1, :]
        Image_b[y, :] = np.sum(patch * filter_y, 0)
    return Image_b

def harris_operator(Image, n_g=5, n_w=5, k=0.06):
    h, w = Image.shape
    sobel_1 = np.array([-1, 0, 1])
    sobel_2 = np.array([1, 2, 1])
    Image_x = calc_derivative_conv(Image, sobel_1, sobel_2)
    Image_y = calc_derivative_conv(Image, sobel_2, sobel_1)
    guassian = get_guassian_kernel(n_g)
    Image_x = calc_derivative_conv(Image_x, guassian, guassian)
    Image_y = calc_derivative_conv(Image_y, guassian, guassian)
    M_temp = np.zeros((h, w, 2, 2))
    M_temp[:, :, 0, 0] = np.square(Image_x)
    M_temp[:, :, 0, 1] = Image_x * Image_y
    M_temp[:, :, 1, 0] = M_temp[:, :, 0, 1]
    M_temp[:, :, 1, 1] = np.square(Image_y)
    guassian = get_guassian_kernel(n_w)
    guassian = np.dstack([guassian] * 4).reshape(n_w, 2, 2)
    M = calc_derivative_conv(M_temp, guassian, guassian)
    P = M[:, :, 0, 0]
    Q = M[:, :, 0, 1]
    R = M[:, :, 1, 1]
    T1 = (P + R) / 2
    T2 = np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2
    Lambda_1 = T1 - T2
    Lambda_2 = T1 + T2
    C = Lambda_1 * Lambda_2 - k * np.square(Lambda_1 + Lambda_2)
    return C, Image_x, Image_y, Lambda_1, Lambda_2

img = array(Image.open("./start.jpeg").convert("L"))
img = (img - img.min())/(img.max()-img.min())
C, I_x, I_y, L_1, L_2 = harris_operator(img)
C = (C - C.min())/(C.max()-C.min())

plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.title('$I_x$')
plt.imshow(I_x, cmap='gray')
plt.subplot(122)
plt.title('$I_y$')
plt.imshow(I_y, cmap='gray')
plt.tight_layout()
plt.show()


plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.title(r'$\lambda_1$')
plt.imshow(L_1, cmap='gnuplot')
plt.subplot(122)
plt.title(r'$\lambda_2$')
plt.imshow(L_2, cmap='gnuplot')
plt.tight_layout()
plt.show()


plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.imshow(C-0.457, cmap='gnuplot')
plt.title('Corner-ness Map')
plt.subplot(122)
plt.imshow(img/2+2*C*(C >= 0.457), cmap='gnuplot')
plt.title('Detected Corners')
plt.tight_layout()
plt.show()
