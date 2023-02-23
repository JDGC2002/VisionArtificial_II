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

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def build_dataset():
    A_images = os.listdir(PATH_A)
    P_images = os.listdir(PATH_P)
    images = []
    labels = []
    for img_name in A_images:
        img_path = os.path.join(PATH_A, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = resize_image(img)
        img = np.ravel(img)
        img = np.array(img)
        img = normalize(img, 0, 1)
        images.append(img)
        labels.append(0)
    for img_name in P_images:
        img_path = os.path.join(PATH_P, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = resize_image(img)
        img = np.ravel(img)
        img = np.array(img)
        img = normalize(img, 0, 1)
        images.append(img)
        labels.append(1)
    df = pd.DataFrame({"imagen": images, "clase": labels})
    return df

def main():
    df = build_dataset()
    df.to_csv("dataset_imagenes.csv", sep=";", index=False, header=False)

if __name__ == '__main__':
    main()