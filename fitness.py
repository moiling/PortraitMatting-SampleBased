import numpy as np


def fitness(f_idx, b_idx, u_idx, f_bgr, b_bgr, u_bgr, f_xy, b_xy, u_xy, u2f_min_dist, u2b_min_dist):
    """
    :param f_idx: foreground samples, type=ndarray, dtype=int64, shape=(f_sample,)
    :param b_idx: background samples, type=ndarray, dtype=int64, shape=(b_sample,)
    :param u_idx: unknown samples,    type=ndarray, dtype=int64, shape=(u_sample,)
    :param f_bgr: all foreground pixels color, type=ndarray, dtype=int64, shape=(f, 3), range=[0, 255]
    :param b_bgr: all background pixels color, type=ndarray, dtype=int64, shape=(b, 3), range=[0, 255]
    :param u_bgr: all unknown pixels color,    type=ndarray, dtype=int64, shape=(u, 3), range=[0, 255]
    :param f_xy:  all foreground pixels coord, type=ndarray, dtype=int64, shape=(f, 2)
    :param b_xy:  all background pixels coord, type=ndarray, dtype=int64, shape=(b, 2)
    :param u_xy:  all unknown pixels coord,    type=ndarray, dtype=int64, shape=(u, 2)
    :param u2f_min_dist: the minimum distance from all unknown pixels to the foreground region.
    :param u2b_min_dist: the minimum distance from all unkonwn pixels to the foreground region.
    :return:
           fit:   fitness per samples,    type=ndarray, dtype=float64, shape=(u, f, b)
           alpha: pred alpha per samples, type=ndarray, dtype=float64, shape=(u, f, b), range=[0, 1]
    """
    u_c = u_bgr[u_idx][:, np.newaxis, np.newaxis, :]  # (u, 1, 1, c=3)
    u_s = u_xy [u_idx][:, np.newaxis, np.newaxis, :]  # (u, 1, 1, c=2)
    f_c = f_bgr[f_idx][np.newaxis, :, np.newaxis, :]  # (1, f, 1, c=3)
    f_s = f_xy [f_idx][np.newaxis, :, np.newaxis, :]  # (1, f, 1, c=2)
    b_c = b_bgr[b_idx][np.newaxis, np.newaxis, :, :]  # (1, 1, b, c=3)
    b_s = b_xy [b_idx][np.newaxis, np.newaxis, :, :]  # (1, 1, b, c=2)

    u2f_md = u2f_min_dist[u_idx][:, np.newaxis, np.newaxis]  # (u, 1, 1)
    u2b_md = u2b_min_dist[u_idx][:, np.newaxis, np.newaxis]  # (u, 1, 1)

    alpha = np.sum((u_c - b_c) * (f_c - b_c), axis=3) / (np.sum((f_c - b_c) ** 2, axis=3) + .01)  # (u, f, b)
    alpha[alpha < 0] = 0
    alpha[alpha > 1] = 1
    alp = alpha[..., np.newaxis]  # (u, f, b, c=1)

    # chromatic cost
    cost_c = np.sqrt(np.sum((u_c - (alp * f_c + (1 - alp) * b_c)) ** 2, axis=3))  # (u, f, b)
    # spatial cost
    cost_sf = np.sqrt(np.sum((f_s - u_s) ** 2, axis=3)) / (u2f_md + .01)  # (u, f, 1)
    cost_sb = np.sqrt(np.sum((b_s - u_s) ** 2, axis=3)) / (u2b_md + .01)  # (u, 1, b)
    fit = cost_c + cost_sf + cost_sb  # (u, f, b)

    return fit, alpha
