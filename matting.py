import cv2
import numpy as np
from fitness import fitness
from scipy.ndimage import distance_transform_edt


def generate_checkerboard(img_shape, step):
    h, w = img_shape
    checkerboard = np.zeros([h, w]).astype(np.bool)
    odd_x, odd_y = np.meshgrid(range(0, img_shape[1], step), range(0, img_shape[0], step))
    even_x, even_y = np.meshgrid(range(1, img_shape[1], step), range(1, img_shape[0], step))
    checkerboard[odd_y, odd_x] = True
    checkerboard[even_y, even_x] = True
    return checkerboard


def matting(img, trimap, max_points=-1):
    """
    :param max_points: maximum samplin points for parallel calculation. if max_points <= f&b sample, no parallel.
    :param img:    input image,    type=ndarray, dtype=uint8, region=[0, 255], shape=(h, w, c), color_type=BGR
    :param trimap: initial trimap, type=ndarray, dtype=uint8, region=[0, 255], shape=(h, w),    color_type=GRAY
    :return:
           alpha_pred: alpha, type=ndarray, dtype=float64, range=[0, 1],   shape=(h, w),    color_type=GRAY
           fg_pred:    fg,    type=ndarray, dtype=uint8,   range=[0, 255], shape=(h, w, c), color_type=BGR
    """
    # get the initial trimap regions.
    f_mask: np.ndarray = trimap > 245
    b_mask: np.ndarray = trimap < 10
    u_mask: np.ndarray = ~f_mask & ~b_mask

    # get f,b,u features.
    f_bgr = img[f_mask].astype(np.int64)
    b_bgr = img[b_mask].astype(np.int64)
    u_bgr = img[u_mask].astype(np.int64)
    f_xy = np.array(np.where(f_mask)).T
    b_xy = np.array(np.where(b_mask)).T
    u_xy = np.array(np.where(u_mask)).T
    # distance to known region.
    u2f_min_dist = distance_transform_edt(~f_mask)[u_mask]
    u2b_min_dist = distance_transform_edt(~b_mask)[u_mask]
    u_size = len(u_bgr)

    # sampling.
    border_mask = cv2.dilate(u_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    f_sample_mask = generate_checkerboard(trimap.shape, 3) & border_mask & f_mask
    b_sample_mask = generate_checkerboard(trimap.shape, 5) & border_mask & b_mask
    [f_idx] = np.where(f_sample_mask[f_mask])
    [b_idx] = np.where(b_sample_mask[b_mask])

    u_alpha = np.zeros(u_size)
    u_fg = np.zeros([u_size, 3])

    cut_size = u_size * len(f_idx) * len(b_idx) / max_points + 1 if max_points > len(f_idx) * len(b_idx) else u_size
    cut_u = np.array_split(range(u_size), cut_size)

    for u_idx in cut_u:
        fit, alpha = fitness(f_idx, b_idx, u_idx, f_bgr, b_bgr, u_bgr, f_xy, b_xy, u_xy, u2f_min_dist, u2b_min_dist)
        best_fb = np.argmin(np.reshape(fit, (len(u_idx), -1)), axis=1)  # (u,)
        best_f = best_fb // len(b_idx)
        best_b = best_fb %  len(b_idx)

        u_alpha[u_idx] = alpha[range(len(u_idx)), best_f, best_b]
        u_fg[u_idx] = f_bgr[f_idx[best_f], :].astype(np.uint8)

    alpha_pred = trimap / 255.
    fg_pred = img.copy()
    alpha_pred[u_mask] = u_alpha
    fg_pred[u_mask] = u_fg

    return alpha_pred, fg_pred

