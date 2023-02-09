import cv2
import numpy as np

img = cv2.imread("cat.jpg")

canal_verde = img[:,:,1]

canal_verde = np.where(canal_verde > 200, 255, canal_verde)

# Reemplazar el canal verde en la img original
img[:,:,1] = canal_verde

# Crear una máscara booleana a partir de la condición anterior
mascara = canal_verde == 255

# Aplicar la máscara al canal rojo y asignar 255 a los valores correspondientes
np.copyto(img[:,:,2], 255, where=mascara)

# Aplicar la máscara al canal azul y asignar 0 a los valores correspondientes
np.copyto(img[:,:,0], 0, where=mascara)

def show_im(img):
    cv2.imshow('Window', img)
    cv2.waitKey(0)
    
show_im(img)