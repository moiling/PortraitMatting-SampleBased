import os
import time

import cv2

from refine_trimap import refine_trimap

if __name__ == '__main__':
    img_dir = 'D:/Mission/photos'
    out_dir = 'out'
    trimap_path = 'D:/Mission/trimap4.png'
    bg_color = [201, 131, 48]  # BGR
    u_min_width = 3  # ensure the unknown region is greater than the minimum width after the trimap refinement.
    f_threshold, b_threshold, = 1., 1.  # f&b confidence threshold of SVM classifier in the trimap refinement.

    os.makedirs(out_dir, exist_ok=True)
    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

    th, tw = trimap.shape

    img_names = os.listdir(img_dir)
    for name in img_names:
        # resize input image.
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

        # refine the initial trimap.
        start_time = time.time()
        trimap_refined = refine_trimap(img_resized, trimap, f_threshold, b_threshold, u_min_width)
        end_time = time.time()
        print(f'refine trimap used {end_time - start_time:.2f}s')

        cv2.imwrite(os.path.join(out_dir, name), trimap_refined)
