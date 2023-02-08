import numpy as np
import cv2 as cv

W = 500
H = 500

def show_im(img):
    cv.imshow('Window', img)
    cv.waitKey(0)

def draw(triangle, image, color=(0,255,0)):
    cv.drawContours(img, [triangle.astype(int)], 0, color, -1)

def translate(triangle, dx, dy):
    t = np.array([dx,dy])
    return triangle +t

def scale(triangle, dx, dy):
    t = np.array([[dx, 0],[0,dy]])
    return triangle @ t

def rotate(triangle, angle):
    t = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return triangle @ t

img = np.zeros((W, H, 3), np.uint8)
triangle = np.array([[10,10], [70,10], [40,60]])
draw(triangle, img)
t1 = translate(triangle, 400, 250)
t2 = scale(triangle, 3, 3)
t3 = rotate(t2, 50)
draw(t3, img, (0,0,255))
show_im(img)
