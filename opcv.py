import cv2 as cv
import numpy as np

s = 100

def imgGen():
    image = np.zeros([100,100])
    for i in range(s):
        image[i,i] = 255
    imgName = input('Ingrese el nombre para la imagen: ')
    cv.imwrite(imgName,image)

imgGen()