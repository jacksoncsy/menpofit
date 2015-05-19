import numpy as np
from copy import deepcopy

def shape_checker(image, shape, fchecker, view=0):
    r"""
    Warp the shape, and use SVM to classify warped image

    Parameters
    ----------
    view : which view to classify; 0 - frontal, 1 - right, 2 - left.
    """

    # convert it to greyscale if needed
    if image.n_channels == 3:
        image = image.as_greyscale(mode='luminosity')

    # get checker for the view
    cker = deepcopy(fchecker[view, 0])

    # get pre-computed data
    nTri  = cker['info_tri'][0, 0][0, 0]
    tri   = cker['tri'][0, 0][0]
    alpha = cker['alpha'][0, 0][0]
    beta  = cker['alpha'][0, 0][0]

    info_mask = cker['info_mask'][0, 0][0]
    mask  = cker['mask'][0, 0][0]
    tridx = cker['tridx'][0, 0][0]
    xmin  = cker['xmin'][0, 0][0]
    ymin  = cker['ymin'][0, 0][0]

    # calc affine coefficient for each triangle
    coeff = CalcCoeff(tri, nTri, shape, alpha, beta)

    # calc warp shape coordinates
    WarpRegion(mask, info_mask, tridx, xmin, ymin, coeff):


def CalcCoeff(tri, nTri, shape, alpha, beta):
    p = shape.points
    coeff = np.zeros(nTri, 6)

    for tIdx in range(nTri):
        i = tri[3*tIdx]
        j = tri[3*tIdx + 1]
        k = tri[3*tIdx + 2]
        c1 = shape[i]
        c2 = shape[j] - c1
        c3 = shape[k] - c1
        c4 = shape[i+p]
        c5 = shape[j+p] - c4
        c6 = shape[k+p] - c4

        tmp_alpha = alpha[3*tIdx : (3*tIdx + 2)]
        tmp_beta  = beta[3*tIdx : (3*tIdx + 2)]

        coeff[tIdx, 0] = c1 + c2*tmp_alpha[0] + c3*tmp_beta[0]
        coeff[tIdx, 1] =      c2*tmp_alpha[1] + c3*tmp_beta[1]
        coeff[tIdx, 2] =      c2*tmp_alpha[2] + c3*tmp_beta[2]
        coeff[tIdx, 3] = c4 + c5*tmp_alpha[0] + c6*tmp_beta[0]
        coeff[tIdx, 4] =      c5*tmp_alpha[1] + c6*tmp_beta[1]
        coeff[tIdx, 5] =      c5*tmp_alpha[2] + c6*tmp_beta[2]

    return coeff

def WarpRegion(mask, info_mask, tridx, xmin, ymin, coeff):
    mapx = np.zeros(info_mask)
    mapy = np.zeros(info_mask)

    for y in range(info_mask[0]):
        yi = y + ymin
        for x in range(info_mask[1]):
            xi = x + xmin
            if mask[y*info_mask[1] + x] == 0:
                mapx[y, x] = -1
                mapy[y, x] = -1