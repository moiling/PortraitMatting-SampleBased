import numpy as np


def compositing(img, fg, bg_color, alpha):
    i = img.astype(np.int64)
    f = fg.astype(np.int64)
    b = np.array(bg_color)
    a = alpha[..., np.newaxis]

    f_weight = .4

    return (1 - f_weight) * a * f + f_weight * a * i + b * (1 - a)
