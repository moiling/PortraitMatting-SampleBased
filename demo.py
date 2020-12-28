import os
import time

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from active_contour import active_contour

if __name__ == '__main__':
    img_dir = 'D:/Mission/photos'
    out_dir = 'out'
    trimap_path = 'D:/Mission/trimap.png'
    bg_color = [201, 131, 48]  # BGR

    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

    bw_f = trimap > 245
    bw_b = trimap < 10
    bw_u = bw_f | bw_b

    img_names = os.listdir(img_dir)
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # [size, 2-channels], int64, 0-255
        bgr_f = img[bw_f].astype(int)
        bgr_b = img[bw_b].astype(int)
        bgr_u = img[bw_u].astype(int)

        start_time = time.time()
        bw_f_refined = bw_f | active_contour(img, bw_f, 50)
        bw_b_refined = bw_b | active_contour(img, bw_b, 50)
        bw_f_refined = binary_fill_holes(bw_f_refined)
        bw_b_refined = binary_fill_holes(bw_b_refined)

        end_time = time.time()
        print(f'active_contour time: {end_time - start_time:.2f}s')

        cv2.imwrite(os.path.join(out_dir, name), bw_b_refined * 255)

    os.makedirs(out_dir, exist_ok=True)
