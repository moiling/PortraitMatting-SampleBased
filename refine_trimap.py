import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from active_contour import active_contour


def refine_trimap(img, trimap, f_threshold=.9, b_threshold=.9, erode_size=2):
    """
    :param erode_size: The kernel size of the last erode operation.
    :param b_threshold: background confidence threshold of SVM classifier. value greater than 0, default=.9
    :param f_threshold: foreground confidence threshold of SVM classifier. value greater than 0, default=.9
    :param img:    input image,    type=ndarray, dtype=uint8, region=[0,255], shape=(h,w,c), color_type=BGR
    :param trimap: initial trimap, type=ndarray, dtype=uint8, region=[0,255], shape=(h,w),   color_type=GRAY
    :return:       refined trimap, type=ndarray, dtype=uint8, region=[0,255], shape=(h,w),   color_type=GRAY
    """
    # get the initial trimap regions.
    f_mask = trimap > 245
    b_mask = trimap < 10
    u_mask = ~f_mask & ~b_mask

    # correct the non-standard trimap.
    trimap[f_mask] = 255
    trimap[b_mask] = 0
    trimap[u_mask] = 128

    # color features.
    img_bgr = img
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # coordinate features.
    h, w, _ = img_bgr.shape
    [img_x, img_y] = np.meshgrid(range(w), range(h))

    img_fea = np.concatenate([
        img_x[..., np.newaxis],
        img_y[..., np.newaxis],
        img_bgr,
        img_hsv,
        img_lab
    ], axis=2)  # [h, w, c=11(xy, bgr, hsv, lab)]

    # segment('active contour' used here) the input image to expend the initial f&b regions to train the SVM model.
    f_mask_refined = f_mask | active_contour(img_bgr, f_mask, 50)
    b_mask_refined = b_mask | active_contour(img_bgr, b_mask, 50)
    f_mask_refined = binary_fill_holes(f_mask_refined)
    b_mask_refined = binary_fill_holes(b_mask_refined)

    # use refined f&b regions as training data for SVM model.
    f_fea = img_fea[f_mask_refined]
    b_fea = img_fea[b_mask_refined]
    u_fea = img_fea[u_mask]
    train_input = np.concatenate([f_fea, b_fea], axis=0).astype(np.int32)
    train_label = np.concatenate([np.ones(len(f_fea)), np.zeros(len(b_fea))]).astype(np.int32)

    # classify the initial unknown region.
    svm_model = make_pipeline(StandardScaler(), LinearSVC(max_iter=1e6))
    svm_model.fit(train_input, train_label)
    u_prob = svm_model.decision_function(u_fea.astype(np.int32))

    # refine the initial trimap.
    trimap_refined = trimap.copy()
    u_label = (np.ones(len(u_fea)) * 128).astype(np.uint8)
    u_label[u_prob > f_threshold] = 255
    u_label[u_prob < -b_threshold] = 0
    trimap_refined[u_mask] = u_label

    # ensure there is an u region in the middle of the f region and b region.
    f_mask_eroded = (trimap_refined == 255).astype(np.uint8)
    f_mask_eroded = cv2.erode(f_mask_eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size)))
    b_mask_eroded = (trimap_refined == 0).astype(np.uint8)
    b_mask_eroded = cv2.erode(b_mask_eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size)))
    trimap_refined[...] = 128
    trimap_refined[f_mask_eroded == 1] = 255
    trimap_refined[b_mask_eroded == 1] = 0

    return trimap_refined
