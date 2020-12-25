import os
import cv2
import numpy as np
import skimage.segmentation as seg

if __name__ == '__main__':
    img_dir = 'D:/Mission/photos'
    out_dir = 'out'
    trimap_path = 'D:/Mission/trimap4.png'

    bg_color = [201, 131, 48]  # BGR

    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    trimap[np.logical_and(trimap > 0, trimap < 255)] = 128

    isf = trimap == 255
    isb = trimap == 0
    isu = np.logical_not(np.logical_or(isf, isb))

    # [y_f, x_f] = np.array(np.where(isf))

    img_names = os.listdir(img_dir)
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # [size, 2-channels], int64, 0-255
        rgb_f = img[isf].astype(int)
        rgb_b = img[isb].astype(int)
        rgb_u = img[isu].astype(int)

        seg.active_contour(img, isf)

    os.makedirs(out_dir, exist_ok=True)
