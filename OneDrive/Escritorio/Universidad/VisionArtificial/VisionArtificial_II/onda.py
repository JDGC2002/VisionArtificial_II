import numpy as np
import cv2

# Crear una imagen de fondo negro
img = np.zeros((500, 500, 3), np.uint8)
img[:,:] = (0, 0, 0)

# Dibujar ondas senoidales en blanco
for i in range(5):
    for x in range(500):
        y = int(250 + 100 * np.sin(x * np.pi / 100 + 2 * np.pi * i / 5))
        img[y, x] = (255, 255, 255)

# Mostrar la imagen
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()