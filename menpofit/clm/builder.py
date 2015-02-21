from __future__ import division, print_function
import numpy as np
from menpo.feature import sparse_hog
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit import checks
from menpofit.base import create_pyramid, build_sampling_grid
from menpofit.builder import (DeformableModelBuilder, build_shape_model,
                              normalization_wrt_reference_shape)
from .classifier import linear_svm_lr

# Shiyang add
from menpo.shape import PointCloud
import menpo.io as mio


class CLMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Constrained Local Models.

    Parameters
    ----------
    classifier_trainers : ``callable -> callable`` or ``[callable -> callable]``

        Each ``classifier_trainers`` is a callable that will be invoked as:

            classifer = classifier_trainer(X, t)

        where X is a matrix of samples and t is a matrix of classifications
        for each sample. `classifier` is then itself a callable,
        which will be used to classify novel instance by the CLM.

        If list of length ``n_levels``, then a classifier_trainer callable is
        defined per level. The first element of the list specifies the
        classifier_trainer to be used at the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified
        classifier_trainer will be used for all levels.


        Examples of such classifier trainers can be found in
        `menpo.fitmultilevel.clm.classifier`

    patch_shape : tuple of `int`
        The shape of the patches used by the classifier trainers.

    patch_size : tuple of `int`
        The size of one patch (positive/negative) to extract patch (for raw/igo feature).

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.


    normalization_diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the ``normalization_diagonal``
        value.
        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames will be the mean shapes of each
        pyramid level, so the shape models will be scaled.

        If ``False``, the reference frames of all levels will be the mean shape
        of the highest level, so the shape models will not be scaled; they will
        have the same size.

    max_shape_components : ``None`` or `int` > ``0`` or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    boundary : `int` >= ``0``, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    Returns
    -------
    clm : :map:`CLMBuilder`
        The CLM Builder object
    """
    def __init__(self, classifier_trainers=linear_svm_lr, patch_shape=(5, 5), patch_size=(5, 5),
                 features=sparse_hog, normalization_diagonal=None,
                 n_levels=3, downscale=1.1, scaled_shape_models=True,
                 max_shape_components=None, boundary=3):

        # general deformable model checks
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        checks.check_boundary(boundary)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)

        # CLM specific checks
        classifier_trainers = check_classifier_trainers(classifier_trainers, n_levels)
        patch_shape = check_patch_shape(patch_shape)

        # Shiyang add
        self.use_scroll_window = check_feature_extraction_type(features)
        self.patch_size = patch_size

        # store parameters
        self.classifier_trainers = classifier_trainers
        self.patch_shape = patch_shape
        self.features = features
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.max_shape_components = max_shape_components
        self.boundary = boundary

    def build(self, images, image_path=None, group=None, label=None, verbose=False):
        r"""
        Builds a Multilevel Constrained Local Model from a list of
        landmarked images.

        Parameters
        ----------
        images : list of :map:`Image`
            The set of landmarked images from which to build the AAM.
        group : string, Optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label : `string`, optional
            The label of of the landmark manager that you wish to use. If
            ``None``, the convex hull of all landmarks is used.
        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        clm : :map:`CLM`
            The CLM object
        """
        # if given data file, then load the data manually
        # Shiayng add, for memory usage concern
        if image_path is not None:
            if len(images) is 0:
                images = []
                for (i, item) in enumerate(image_path):
                    images.extend(mio.import_pickle(item))
            else:
                ValueError("Should not pass both images and image_path!!")
        else:
            if len(images) is 0:
                ValueError("No data provided!!")


        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            normalization_wrt_reference_shape(
                images, group, label, self.normalization_diagonal,
                verbose=verbose)

        # Shiayng add
        if image_path is not None:
            del images

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)
        # Shiyang add
        n_training_images = len(normalized_images)
        del normalized_images

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        classifiers = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters of type list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get images of current level
            feature_images = []
            for c, g in enumerate(generators):
                if verbose:
                    print_dynamic(
                        '{}Computing feature space/rescaling - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = build_shape_model(
                train_shapes, self.max_shape_components[rj])

            # add shape model to the list
            shape_models.append(shape_model)

            # build classifiers
            sampling_grid = build_sampling_grid(self.patch_shape)
            n_points = shapes[0].n_points
            level_classifiers = []
            for k in range(n_points):
                if verbose:
                    print_dynamic('{}Building classifiers - {}'.format(
                        level_str,
                        progress_bar_str((k + 1.) / n_points,
                                         show_bar=False)))

                positive_labels = []
                negative_labels = []
                positive_samples = []
                negative_samples = []

                for i, s in zip(feature_images, shapes):

                    max_x = i.shape[0] - 1
                    max_y = i.shape[1] - 1

                    point = (np.round(s.points[k, :])).astype(int)
                    patch_grid = sampling_grid + point[None, None, ...]
                    positive, negative = get_pos_neg_grid_positions(
                        patch_grid, positive_grid_size=(1, 1))

                    x = positive[:, 0]
                    y = positive[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    # Shiyang add
                    if self.use_scroll_window:
                        positive_sample = extract_scroll_window(i, x, y, self.patch_size)
                    else:
                        positive_sample = i.pixels[:, x, y].T

                    positive_samples.append(positive_sample)
                    positive_labels.append(np.ones(positive_sample.shape[0]))

                    x = negative[:, 0]
                    y = negative[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    # Shiyang add
                    if self.use_scroll_window:
                        negative_sample = extract_scroll_window(i, x, y, self.patch_size)
                    else:
                        negative_sample = i.pixels[:, x, y].T

                    negative_samples.append(negative_sample)
                    negative_labels.append(-np.ones(negative_sample.shape[0]))

                positive_samples = np.asanyarray(positive_samples)
                positive_samples = np.reshape(positive_samples,
                                              (-1, positive_samples.shape[-1]))
                positive_labels = np.asanyarray(positive_labels).flatten()

                negative_samples = np.asanyarray(negative_samples)
                negative_samples = np.reshape(negative_samples,
                                              (-1, negative_samples.shape[-1]))
                negative_labels = np.asanyarray(negative_labels).flatten()

                X = np.vstack((positive_samples, negative_samples))
                t = np.hstack((positive_labels, negative_labels))

                clf = self.classifier_trainers[rj](X, t)
                level_classifiers.append(clf)

            # add level classifiers to the list
            classifiers.append(level_classifiers)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        classifiers.reverse()

        from .base import CLM
        return CLM(shape_models, classifiers, n_training_images,
                   self.patch_shape, self.patch_size, self.features, self.reference_shape,
                   self.downscale, self.scaled_shape_models, self.use_scroll_window)


def get_pos_neg_grid_positions(sampling_grid, positive_grid_size=(1, 1)):
    r"""
    Divides a sampling grid in positive and negative pixel positions. By
    default only the centre of the grid is considered to be positive.
    """
    positive_grid_size = np.array(positive_grid_size)
    mask = np.zeros(sampling_grid.shape[:-1], dtype=np.bool)
    centre = np.round(np.array(mask.shape) / 2).astype(int)
    positive_grid_size -= [1, 1]
    start = centre - positive_grid_size
    end = centre + positive_grid_size + 1
    mask[start[0]:end[0], start[1]:end[1]] = True
    positive = sampling_grid[mask]
    negative = sampling_grid[~mask]
    return positive, negative


def check_classifier_trainers(classifier_trainers, n_levels):
    r"""
    Checks the classifier_trainers. Must be a ``callable`` ->
    ``callable`` or
    or a list containing 1 or {n_levels} callables each of which returns a
    callable.
    """
    str_error = ("classifier must be a callable "
                 "of a list containing 1 or {} callables").format(n_levels)
    if not isinstance(classifier_trainers, list):
        classifier_list = [classifier_trainers] * n_levels
    elif len(classifier_trainers) == 1:
        classifier_list = [classifier_trainers[0]] * n_levels
    elif len(classifier_trainers) == n_levels:
        classifier_list = classifier_trainers
    else:
        raise ValueError(str_error)
    for classifier in classifier_list:
        if not callable(classifier):
            raise ValueError(str_error)
    return classifier_list


def check_patch_shape(patch_shape):
    r"""
    Checks the patch shape. It must be a tuple with `int` > ``1``.
    """
    str_error = "patch_size mast be a tuple with two integers"
    if not isinstance(patch_shape, tuple) or len(patch_shape) != 2:
        raise ValueError(str_error)
    for sh in patch_shape:
        if not isinstance(sh, int) or sh < 2:
            raise ValueError(str_error)
    return patch_shape


# Shiyang add
def check_feature_extraction_type(features):
    r"""
    Checks the feature type. If is raw/igo, then need to apply scrolling window
    """
    feature_names = ['no_op', 'igo']
    use_scroll_window = False

    for (i, item) in enumerate(feature_names):
        if item == features.func_name:
            use_scroll_window = True

    return use_scroll_window


# Shiyang add
def extract_scroll_window(image, x, y, patch_size):
    r"""
    If is raw/igo, then need to apply scrolling window

    image : Image
        image to extract feature

    x, y : ndarray of `int`
        x and y coordinates of point centers

    """
    tmp = np.float64(np.concatenate((x[:, None], y[:, None]), axis=1))
    point_centers = PointCloud(tmp)
    sample_list = image.extract_patches(point_centers, patch_size)
    flatten_sample_list = []
    for (idx, item) in enumerate(sample_list):
        b = item.as_vector()[None, :]
        flatten_sample_list.append(b)

    flatten_sample_list = np.asanyarray(flatten_sample_list)
    sample = np.reshape(flatten_sample_list, (-1, flatten_sample_list.shape[-1]))

    return sample