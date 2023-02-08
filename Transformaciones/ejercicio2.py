import cv2 as cv

imgF = cv.imread('test_im.jpg')
imga = cv.imread('test_im-A.jpg')
imgB = cv.imread('test_im-B.jpg')

mascara1 = imga[:,:,0] <= 20
mascara2 = imga[:,:,1] <= 20
mascara3 = imga[:,:,2] <= 20

np.copyto(imgF[:,:,0], imgB[:,:,0], where=mascara1)
np.copyto(imgF[:,:,1], imgB[:,:,1], where=mascara2)
np.copyto(imgF[:,:,2], imgB[:,:,2], where=mascara3)

def show_im(img):
    cv.imshow('Window', img)
    cv.waitKey(0)

show_im(imgF)