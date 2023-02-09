import cv2 as cv
import numpy as np

imgcat = cv.imread('test_im.jpg')
dia = cv.imread('day.jpg')
noche = cv.imread('night.jpg')

def show_im(img):
    cv.imshow('Window', img)
    cv.waitKey(0)
    
def brillo_contraste(img, gain, bias):
    return (img * gain) + bias

def mezcla_lineal(img1, img2, alpha):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ((1-alpha)*img1) + (img2*alpha)

def correccion_gamma(image, gamma=1.0):
    gamma_corr = image / 255.0
    gamma_corr = np.power(gamma_corr, gamma)
    gamma_corr = np.uint8(gamma_corr * 255)

    return gamma_corr


brillo_contras = brillo_contraste(imgcat, 1.5, 0.2)
mezcla_lin = mezcla_lineal(dia, noche, 0)
correcion_gam = correccion_gamma(imgcat, 2)
show_im(brillo_contras)
show_im(mezcla_lin)
show_im(correcion_gam)