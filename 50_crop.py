import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import SimpleITK as sitk
import os

folder = r"C:\Users\difrischiamm\Desktop\fractal\data\1_cropmaskssub"
#folder = r"C:\Users\difrischiamm\Desktop\HAM10000_images_part_1"


def tight_crop(mask):
    print(mask.shape)
    mask = mask.astype(np.uint8)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ys, xs = np.where(thr > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = thr[y_min:y_max+1, x_min:x_max+1]
    padded = np.pad(cropped, pad_width=50, mode='constant', constant_values=0)
    return padded

smallest = 1e8
smol_file = ''
size = ()

for i in os.listdir(folder):
    total_path = os.path.join(folder, i)
    output_path = os.path.join(folder, i, f"{i}_cropped")
    os.makedirs(output_path, exist_ok=True)
    for image in os.listdir(total_path):
        print(image)
        filepath = os.path.join(total_path, image)
        if os.path.isdir(filepath):
            continue
        img = sitk.ReadImage(filepath)
        arr = sitk.GetArrayFromImage(img)
        cropped_arr = tight_crop(arr)
        crepe = arr.shape
        area = crepe[0] * crepe[1]
        print(f"Area: {area}")
        if area < smallest:
            smallest = area
            size = arr.shape
            smol_file = image
        cv2.imwrite(output_path + f'/{image}.jpg', cropped_arr)

print('*****************************************')
print(f"Smallest area: {smallest} which was size {size} from {smol_file}")
print('*****************************************')