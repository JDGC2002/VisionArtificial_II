import os
import cv2
import pandas as pd
import numpy as np

PATH_A = "dataset/A"
PATH_P = "dataset/P"

def charge_dir(path):
    dir_list = os.listdir(path)
    return "\n".join(["Files and directories in '{}':".format(path)] + dir_list)

def resize_image(image):
    return np.resize(image, (32,32))

def build_dataset():
    A_images = os.listdir(PATH_A)
    P_images = os.listdir(PATH_P)
    images = []
    labels = []
    for img_name in A_images:
        img_path = os.path.join(PATH_A, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = resize_image(img)
        img = np.array(img)
        images.append(img)
        labels.append(0)
    for img_name in P_images:
        img_path = os.path.join(PATH_P, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = resize_image(img)
        img = np.array(img)
        images.append(img)
        labels.append(1)
    df = pd.DataFrame({"imagen": images, "clase": labels})
    return df

def main():
    df = build_dataset()
    df.to_csv("dataset_imagenes.csv", sep=";", index=False, header=False)

if __name__ == '__main__':
    main()