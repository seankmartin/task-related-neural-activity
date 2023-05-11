from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import numpy as np
import similaritymeasures


def procrustes_modify(X, Y, orthogonal=False):
    if orthogonal:
        R, scale = orthogonal_procrustes(X, Y)
        result = np.dot(X, R), Y
    else:
        mtx1, mtx2, disparity = procrustes(X, Y)
        result = mtx1, mtx2
    return result


def distance_between_curves(X, Y):
    return similaritymeasures.frechet_dist(X, Y)
