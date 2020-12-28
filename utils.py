import cv2
import numpy as np


def mask2contours_rc(mask):
    """
    :param mask:
    :return: rc(row column) coordinates.
    """
    idx = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0]
    contours_rc = np.array([idx[:, 0, 1], idx[:, 0, 0]]).T
    return contours_rc
