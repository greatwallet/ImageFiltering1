# image filter assignment:

# tilt shift / motion blur

# by cxt FR Tsinghua 

import numpy as np
import time
from skimage import io, data
from math import *
import sys

filename = ""
# convolution operation (one channel only)
def conv2d(f,kernel):
    width, length = f.shape
    size = kernel.shape[0]
    # zero padding
    padding = int((size-1)/2)
    f = np.pad(f,padding, 'constant')
    # rotate 180 degree
    kernel = np.rot90(kernel,2)
    # multiply and add 
    g = np.zeros((width,length),dtype = f.dtype)
    for i in range(width):
        for j in range(length):
            for k in range(size):
                for h in range(size):
                    g[i][j] += int(f[i+k][j+h]*kernel[k][h])
    return g

# generate gaussian function f(x) (not regularized)
# condition: f(mu) = 1
def gaussian_function(x,mu,sigma):
    return exp(-pow(x-mu,2)/2*pow(sigma,2))

# generate mask for tilt shift
def generate_mask(img, sigma = 3e-2):
    mask = np.zeros(img.shape, dtype = img.dtype)
    r, c = mask.shape
    for i in range(r):
        if mask.dtype == 'int':
            mask[i, :] = int(255 * gaussian_function(i, r/2,sigma))
        else:
            mask[i, :] = 255 * gaussian_function(i, r/2,sigma)
    return mask

# generate gaussian_filter of shape(5,5)
def generate_gaussian_filter():
    gaussian_filter = np.array([1,4,6,4,1])/16
    gaussian_filter = np.reshape(gaussian_filter,(gaussian_filter.shape[0],-1))
    gaussian_filter = np.transpose(gaussian_filter)*gaussian_filter
    return gaussian_filter

# generate gaussian blurred image
def generate_gaussian_blurred_img(img, k = 5):
    print("Start generating gaussian blurred image!")
    gaussian_filter = generate_gaussian_filter()
    blurred_img = img
    print("You will have to wait for %d seconds for the blurred image done" %(k*29))
    for i in range(k):
        blurred_img = conv2d(blurred_img,gaussian_filter)
    return blurred_img

# generate tilt-shift effect on the image
def tilt_shift(img, sigma = 3e-2, k = 5):
    r,c = img.shape
    img_tf = np.zeros(img.shape, dtype = img.dtype)
    mask = generate_mask(img,sigma)
    blurred_img = generate_gaussian_blurred_img(img,k)
    print("Start tilt shifting!")
    for i in range(r):
        for j in range(c):
            alpha = mask[i, j] / 255
            img_tf[i, j] = img[i ,j]*alpha + blurred_img[i, j]*(1-alpha)
    io.imsave("tilt shift " + filename,img_tf)
    return img_tf[i, j]

# generate spin blurred image
def spin_blur(img):
    spin_img = img.copy()
    row, col= img.shape
    xx = np.arange (col)
    yy = np.arange (row)
    x_mask = np.matlib.repmat (xx, row, 1)
    y_mask = np.matlib.repmat (yy, col, 1)
    y_mask = np.transpose(y_mask)
    center_y = (row -1) / 2.0
    center_x = (col -1) / 2.0
    R = np.sqrt((x_mask - center_x) **2 + (y_mask - center_y) ** 2)
    angle = np.arctan2(y_mask - center_y , x_mask - center_x)
    Num = 20
    arr = ( np.arange(Num) + 1 ) / 100.0
    for i in range (row):
        for j in range (col):
            T_angle = angle[i, j] + arr
            new_x = R[i, j] * np.cos(T_angle) + center_x
            new_y = R[i, j] * np.sin(T_angle) + center_y
            int_x = new_x.astype(int)
            int_y = new_y.astype(int)
            int_x[int_x > col-1] = col - 1
            int_x[int_x < 0] = 0
            int_y[int_y < 0] = 0
            int_y[int_y > row -1] = row -1
            spin_img[i,j] = img[int_y, int_x].sum()/Num
    return spin_img

# generate motion blurred image: including:
#   spin blur
#   x-axis blur
#   y-axis blur
def motion_blur(img, COMMAND):
    if COMMAND == "-x":
        x_blur_filter = np.zeros((5,5))
        x_blur_filter[2,:] = 1/5
        img_x = conv2d(img,x_blur_filter)
        io.imsave("x blur " + filename, img_x)
    elif COMMAND == "-y":
        y_blur_filter = np.zeros((5,5))
        y_blur_filter[:,2] = 1/5
        img_y = conv2d(img,y_blur_filter)
        io.imsave("y blur " + filename,img_y)
    elif COMMAND == "-sp":
        spin_img = spin_blur(img)
        io.imsave("spin blur " + filename ,spin_img)
    else:
        print("Error in your commands!")
        return 

if __name__ == "__main__":
    print("HINT: check \"python imageFilter.py -h\" for help")
    filename = sys.argv[1]
    if filename == "-h":
        print("usage: python imageFilter.py [image file] [command]")
        print("Commands: ")
        print("-x       : generate X-AXIS MOTION BLURRED image")
        print("-y       : generate Y-AXIS MOTION BLURRED image")
        print("-sp      : generate SPIN BLURRED image")
        print("-tf      : generate TILT-SHIFT image")
        print("-h       : print this help message and exit")
    if len(sys.argv) < 3:
        sys.exit(0)
    COMMAND = sys.argv[2]
    img = io.imread(filename)
    # operate on ONE-CHANNELED image only
    if len(img.shape) > 2:
        img = img[:,:,0]
    if COMMAND == "-tf":
        print("The process may take some time, please wait patiently.")
        tilt_shift(img)
        print("The output image has been saved!")
    elif COMMAND == "-h ":
        print("usage: python imageFilter.py [image file] [command]")
        print("Commands: ")
        print("-x       : generate X-AXIS MOTION BLURRED image")
        print("-y       : generate Y-AXIS MOTION BLURRED image")
        print("-sp      : generate SPIN BLURRED image")
        print("-tf      : generate TILT-SHIFT image")
        print("-h       : print this help message and exit")
    else:
        print("The process may take some time, please wait patiently.")
        motion_blur(img, COMMAND)
        print("The output image has been saved!")