import os
import time

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from active_contour import active_contour

if __name__ == '__main__':
    img_dir = 'D:/Mission/photos'
    out_dir = 'out'
    trimap_path = 'D:/Mission/trimap.png'
    bg_color = [201, 131, 48]  # BGR

    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

    bw_f = trimap > 245
    bw_b = trimap < 10
    bw_u = ~bw_f & ~bw_b

    img_names = os.listdir(img_dir)
    for name in img_names:
        # image features.
        img_bgr = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        h, w, _ = img_bgr.shape

        [img_x, img_y] = np.meshgrid(range(w), range(h))
        # [h, w, c=11(xy,bgr,hsv,lab)]
        img_fea = np.concatenate([img_x[..., np.newaxis], img_y[..., np.newaxis], img_bgr, img_hsv, img_lab], axis=2)

        # use segmentation to refined initial f/b region for training SVM.
        start_time = time.time()
        bw_f_refined = bw_f | active_contour(img_bgr, bw_f, 50)
        bw_b_refined = bw_b | active_contour(img_bgr, bw_b, 50)
        bw_f_refined = binary_fill_holes(bw_f_refined)
        bw_b_refined = binary_fill_holes(bw_b_refined)

        end_time = time.time()
        print(f'active_contour time: {end_time - start_time:.2f}s')

        # create SVM train data.
        f_fea = img_fea[bw_f_refined]
        b_fea = img_fea[bw_b_refined]
        u_fea = img_fea[bw_u]
        train_input = np.concatenate([f_fea, b_fea], axis=0).astype(np.int32)
        train_label = np.concatenate([np.ones(len(f_fea)), np.zeros(len(b_fea))]).astype(np.int32)

        # use SVM to refined initial trimap.
        start_time = time.time()
        svm_model = make_pipeline(StandardScaler(), LinearSVC(max_iter=1e6))
        svm_model.fit(train_input, train_label)
        end_time = time.time()
        print(f'SVM fit time: {end_time - start_time:.2f}s')

        u_prob = svm_model.decision_function(u_fea.astype(np.int32))

        f_threshold, b_threshold = .9, -.9

        # create new trimap
        trimap_refined = trimap.copy()
        u_label = (np.ones(len(u_fea)) * 128).astype(np.uint8)
        u_label[u_prob > f_threshold] = 255
        u_label[u_prob < b_threshold] = 0
        trimap_refined[bw_u] = u_label

        cv2.imwrite(os.path.join(out_dir, name), trimap_refined)

    os.makedirs(out_dir, exist_ok=True)
