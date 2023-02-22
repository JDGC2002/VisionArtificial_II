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
    return cv2.resize(image, (32,32))

def build_dataset():
    A_images = os.listdir(PATH_A)
    P_images = os.listdir(PATH_P)
    data = []
    for img_name in A_images:
        img_path = os.path.join(PATH_A, img_name)
        img = cv2.imread(img_path)
        img = np.array(img)
        img = resize_image(img)
        data.append((img, 0))
    for img_name in P_images:
        img_path = os.path.join(PATH_P, img_name)
        img = cv2.imread(img_path)
        img = np.array(img)
        img = resize_image(img)
        data.append((img, 1))
    df = pd.DataFrame(data, columns=["image", "label"])
    return df

def main():
    df = build_dataset()
    df.to_csv("dataset_imagenes.csv", index=False)

if __name__ == '__main__':
    main()