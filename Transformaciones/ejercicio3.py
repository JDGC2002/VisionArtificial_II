import cv2
import numpy as np

# Load the image
img = cv2.imread("cat.jpg")

# Seleccionar el canal verde
canal_verde = img[:,:,1]

# Aplicar la condición y asignar 0 a los valores mayores a 200
canal_verde = np.where(canal_verde > 200, 0, canal_verde)

# Reemplazar el canal verde en la img original
img[:,:,1] = canal_verde

# Crear una máscara booleana a partir de la condición anterior
mascara = canal_verde == 0

# Aplicar la máscara al canal azul y asignar 255 a los valores correspondientes
np.copyto(img[:,:,2], 255, where=mascara)

# Aplicar la máscara al canal rojo y asignar 0 a los valores correspondientes
np.copyto(img[:,:,0], 0, where=mascara)

def show_im(img):
    cv2.imshow('Window', img)
    cv2.waitKey(0)
    
show_im(img)