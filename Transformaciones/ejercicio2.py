import cv2 as cv

imgF = cv.imread('test_im.jpg')
imga = cv.imread('test_im-A.jpg')
imgB = cv.imread('test_im-B.jpg')

def show_im(img):
    cv.imshow('Window', img)
    cv.waitKey(0)

def imageMatting():
    return (1-imga)*imgB + imga*imgF

imgC = imageMatting()
show_im(imgC)