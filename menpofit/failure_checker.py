import numpy as np
from copy import deepcopy
from scipy.interpolate import interp2d, RectBivariateSpline

def shape_checker(image, shape, fchecker, view=0):
    r"""
    Warp the image according to the shape, and use SVM to classify face image

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
    beta  = cker['beta'][0, 0][0]

    info_mask = cker['info_mask'][0, 0][0]
    mask  = cker['mask'][0, 0][0]
    tridx = cker['tridx'][0, 0][0]
    xmin  = cker['xmin'][0, 0][0]
    ymin  = cker['ymin'][0, 0][0]

    weights  = cker['weights'][0, 0][0]
    bias     = cker['bias'][0, 0][0]

    # calc affine coefficient for each triangle
    coeff = CalcCoeff(tri, nTri, shape, alpha, beta)

    # calc warp shape coordinates
    mapx, mapy = WarpRegion(mask, info_mask, tridx, xmin, ymin, coeff)

    # warping
    x = np.arange(0, image.shape[0])
    y = np.arange(0, image.shape[1])
    f = RectBivariateSpline(x, y, image.pixels[0])
    warp = f.ev(mapx, mapy)

    vec_warp = []
    for i, (pixel, m) in enumerate(zip(warp.flatten(), mask)):
        if m == 1:
            vec_warp.append(pixel)

    # center the vector and normalize
    vec_warp = vec_warp - np.mean(vec_warp)
    var = vec_warp.T.dot(vec_warp)
    if var < 1e-10:
        vec = vec_warp * 0.0;
    else:
        vec = vec_warp/np.sqrt(var)

    # prediction
    score = weights.dot(vec) + bias

    return score, warp


def CalcCoeff(tri, nTri, shape, alpha, beta):
    coeff = np.zeros((nTri, 6))

    for tIdx in range(nTri):
        i = tri[3*tIdx]
        j = tri[3*tIdx + 1]
        k = tri[3*tIdx + 2]
        c1 = shape.points[i, 0]
        c2 = shape.points[j, 0] - c1
        c3 = shape.points[k, 0] - c1
        c4 = shape.points[i, 1]
        c5 = shape.points[j, 1] - c4
        c6 = shape.points[k, 1] - c4

        tmp_alpha = alpha[3*tIdx : (3*tIdx + 2) + 1]
        tmp_beta  = beta[3*tIdx : (3*tIdx + 2) + 1]

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
            else:
                tIdx = tridx[y*info_mask[1] + x]
                tmp_coeff = coeff[tIdx, :]
                mapx[y, x] = tmp_coeff[0] + tmp_coeff[1]*xi + tmp_coeff[2]*yi
                mapy[y, x] = tmp_coeff[3] + tmp_coeff[4]*xi + tmp_coeff[5]*yi

    return mapx, mapy

def response_checker(clm, image, shape, level=-1):
    r"""
    Check the response of current shape, and return score of every landmark.

    Parameters
    -----------
    level: `int`, optional
        The pyramidal level to be used.
    """
    # convert it to greyscale if needed
    if image.n_channels == 3:
        image = image.as_greyscale(mode='luminosity')

    # attach landmarks to the image
    image.landmarks['target_shape'] = shape

    # rescale image
    image = image.rescale_to_reference_shape(clm.reference_shape, group='target_shape')

    # apply pyramid
    if clm.n_levels > 1:
        if clm.pyramid_on_features:
            # compute features at highest level
            feature_image = clm.features(image)

            # apply pyramid on feature image
            pyramid = feature_image.gaussian_pyramid(
                n_levels=clm.n_levels, downscale=clm.downscale)

            # get rescaled feature images
            images = list(pyramid)
        else:
            # create pyramid on intensities image
            pyramid = image.gaussian_pyramid(
                n_levels=clm.n_levels, downscale=clm.downscale)

            # compute features at each level
            images = [clm.features[clm.n_levels - j - 1](i)
                      for j, i in enumerate(pyramid)]
        images.reverse()
    else:
        images = [clm.features(image)]

    # initialize responses
    level_image = images[level]
    level_classifiers = clm.classifiers[level]
    level_shape = level_image.landmarks['target_shape'].lms

    max_h = level_image.shape[0] - 1
    max_w = level_image.shape[1] - 1
    response_data = np.zeros((1, clm.n_classifiers_per_level[level]))
    for j, (p, clf) in enumerate(zip(level_shape.points, level_classifiers)):
        pixel = np.round(p).astype(int)

        # deal with boundary
        if pixel[0] > max_h:
            pixel[0] = max_h
        elif pixel[0] < 0:
            pixel[0] = 0

        if pixel[1] > max_w:
            pixel[1] = max_w
        elif pixel[1] < 0:
            pixel[1] = 0

        # get response
        response_data[:, j] = clf(level_image.pixels[:, pixel[0], pixel[1]])

    return response_data