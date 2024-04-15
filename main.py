# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:31:48 2024

@author: zheng
"""

import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal



def Dot(a, b):
    res = (a[:, :, 0] * b[:, :, 0]) + (a[:, :, 1] * b[:, :, 1]) + (a[:, :, 2] * b[:, :, 2])
    return res.reshape(res.shape[0], res.shape[1], 1)



def Normalize(arr):
    lens = ((arr[:, :, 0] ** 2) + (arr[:, :, 1] ** 2) + (arr[:, :, 2] ** 2)) ** 0.5
    return arr/lens.reshape(lens.shape[0], lens.shape[1], 1);



def MakeFilter(width:int, height:int):
    arr = np.asarray(Image.open('filter2.png'), 'float32') - 128
    
    #to normal map
    normal = Normalize(arr)
    
    #ajust size
    tileY = math.ceil(height/normal.shape[0])
    tileX = math.ceil(width/normal.shape[1])
    
    return np.tile(normal, (tileY, tileX, 1))[0: height, 0: width, :]
    
 
    
def ComputeRefraction(normImg):
    index0 = 1 #air
    index1 = 1.3333 #water
    raysIn = np.zeros(normImg.shape, 'float32')
    
    #rays go toward to the screen
    raysIn[:, :, 2] = -1
    
    #snell's law in vector form
    dn = Dot(raysIn, normImg)
    raysOut0 = (raysIn - (normImg * dn)) * (index0/index1)
    raysOut1 = normImg * ((1 - (((index0 ** 2) / (index1 ** 2)) * (1 - (dn ** 2)))) ** 0.5)
    
    return Normalize(raysOut0 - raysOut1)

 

def Mapping(rays, img, distance):
    
    #add depth
    depths = (distance / (-1 * rays[:, :, 2]))
    raysWithDepths = rays[:, :, :]
    
    raysWithDepths[:, :, 0] = raysWithDepths[:, :, 0] * depths
    raysWithDepths[:, :, 1] = raysWithDepths[:, :, 1] * depths
    raysWithDepths[:, :, 2] = raysWithDepths[:, :, 2] * depths
    
    
    #gen final img
    outImg = np.zeros(img.shape)
    black = np.zeros(3)
    
    for y in range(0, outImg.shape[0]):
        for x in range(0, outImg.shape[1]):
            mappedX = x + rays[y, x, 0]
            mappedY = y + rays[y, x, 1]
            
            if(mappedY < 0 or mappedY >= img.shape[0] or mappedX < 0 or mappedX >= img.shape[1]):
                outImg[y, x, :] = black
            else:
                outImg[y, x, :] = img[math.floor(mappedY), math.floor(mappedX), :]
            
    
    return outImg
    


def GaussianFilter(kernelRad: int):
    
    f = np.zeros((kernelRad * 2 + 1, kernelRad * 2 + 1))
    sigma = 2.5 * kernelRad
    
    for x in range(-kernelRad, kernelRad + 1):
        for y in range(-kernelRad, kernelRad + 1):
            f[kernelRad + x, kernelRad + y] = (1/((2*math.pi*(sigma**2))**0.5)) * math.exp(-(((x**2) + (y**2))/(2*(sigma**2))))
    
    return f/np.sum(f)



def GaussianBlur(img):
    
    kernel = GaussianFilter(10)
    resR = signal.convolve2d(img[:, :, 0], kernel, mode='valid')
    resG = signal.convolve2d(img[:, :, 1], kernel, mode='valid')
    resB = signal.convolve2d(img[:, :, 2], kernel, mode='valid')
    
    retImg = np.zeros((resR.shape[0], resR.shape[1], 3))
    retImg[:, :, 0] = resR[:, :]
    retImg[:, :, 1] = resG[:, :]
    retImg[:, :, 2] = resB[:, :]
    
    return retImg
    


#read image
img = np.asarray(Image.open('test.jpg'))/255

#to blur image
blrImg = GaussianBlur(img)


#read filter
f = MakeFilter(blrImg.shape[1], blrImg.shape[0])

#compute rays
raysOut = ComputeRefraction(f)

#compute final image
finalImg = Mapping(raysOut, blrImg, 500)


plt.figure(figsize = (20,20))
plt.imshow(finalImg)
plt.show()


