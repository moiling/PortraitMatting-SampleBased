import cv2
import skimage.segmentation as seg


def active_contour(img, mask, iteration):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segment = seg.morphological_chan_vese(img_grey, iteration, mask)
    return segment
