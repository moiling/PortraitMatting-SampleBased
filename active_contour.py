import skimage.segmentation as seg
import cv2


def active_contour(img, mask, iteration):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    igg_img = seg.inverse_gaussian_gradient(img_grey)
    segment = seg.morphological_geodesic_active_contour(igg_img, iteration, mask, smoothing=1, balloon=1, threshold=0.7)
    return segment
