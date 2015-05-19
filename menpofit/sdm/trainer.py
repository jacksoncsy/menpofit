from __future__ import division, print_function
import abc
import numpy as np
from menpo.transform import Scale, AlignmentSimilarity
from menpo.shape import mean_pointcloud
from menpo.feature import sparse_hog, no_op
from menpofit.modelinstance import PDM, OrthoPDM
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit import checks
from menpofit.transform import (ModelDrivenTransform, OrthoMDTransform,
                                DifferentiableAlignmentSimilarity)
from menpofit.regression.trainer import (
    NonParametricRegressorTrainer, ParametricRegressorTrainer,
    SemiParametricClassifierBasedRegressorTrainer)
from menpofit.regression.regressors import mlr, rls
from menpofit.regression.parametricfeatures import weights
from menpofit.base import DeformableModel, create_pyramid
from .fitter import SDMFitter, SDAAMFitter, SDCLMFitter

# Shiyang add
from menpofit.builder import build_shape_model
from copy import deepcopy
import menpo.io as mio
import dill


def check_regression_features(regression_features, n_levels):
    try:
        return checks.check_list_callables(regression_features, n_levels)
    except ValueError:
        raise ValueError("regression_features must be a callable or a list of "
                         "{} callables".format(n_levels))


def check_regression_type(regression_type, n_levels):
    r"""
    Checks the regression type (method) per level.

    It must be a callable or a list of those from the family of
    functions defined in :ref:`regression_functions`

    Parameters
    ----------
    regression_type : `function` or list of those
        The regression type to check.

    n_levels : `int`
        The number of pyramid levels.

    Returns
    -------
    regression_type_list : `list`
        A list of regression types that has length ``n_levels``.
    """
    try:
        return checks.check_list_callables(regression_type, n_levels)
    except ValueError:
        raise ValueError("regression_type must be a callable or a list of "
                         "{} callables".format(n_levels))


def check_n_permutations(n_permutations):
    if n_permutations < 1:
        raise ValueError("n_permutations must be > 0")


def apply_pyramid_on_images(generators, n_levels, verbose=False):
    r"""
    Exhausts the pyramid generators verbosely
    """
    all_images = []
    for j in range(n_levels):

        if verbose:
            level_str = '- Apply pyramid: '
            if n_levels > 1:
                level_str = '- Apply pyramid: [Level {} - '.format(j + 1)

        level_images = []
        for c, g in enumerate(generators):
            if verbose:
                print_dynamic(
                    '{}Computing feature space/rescaling - {}'.format(
                        level_str,
                        progress_bar_str((c + 1.) / len(generators),
                                         show_bar=False)))
            level_images.append(next(g))
        all_images.append(level_images)
    if verbose:
        print_dynamic('- Apply pyramid: Done\n')
    return all_images


class SDTrainer(DeformableModel):
    r"""
    Mixin for Supervised Descent Trainers.

    Parameters
    ----------
    regression_type : `callable`, or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        Examples of such callables can be found in :ref:`regression_callables`.
    regression_features :`` None`` or `callable` or `[callable]`, optional
        The features that are used during the regression.

        If `list`, a regression feature is defined per level.

        If not list or list with length ``1``, the specified regression feature
        will be used for all levels.

        Depending on the :map:`SDTrainer` object, this parameter can take
        different types.
    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.
    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.
    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)
    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.
    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.

    Returns
    -------
    fitter : :map:`MultilevelFitter`
        The fitter object.

    Raises
    ------
    ValueError
        ``regression_type`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    ValueError
        n_levels must be `int` > ``0``
    ValueError
        ``downscale`` must be >= ``1``
    ValueError
        ``n_perturbations`` must be > 0
    ValueError
        ``features`` must be a `string` or a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, regression_type=mlr, regression_features=None,
                 features=no_op, n_levels=3, n_iterations=10, downscale=1.2,
                 noise_std=0.04, rotation=False, n_perturbations=10):
        features = checks.check_features(features, n_levels)
        DeformableModel.__init__(self, features)

        # general deformable model checks
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)

        # SDM specific checks
        regression_type_list = check_regression_type(regression_type,
                                                     n_levels)
        regression_features = check_regression_features(regression_features,
                                                        n_levels)
        check_n_permutations(n_perturbations)

        # store parameters
        self.regression_type = regression_type_list
        self.regression_features = regression_features
        self.n_levels = n_levels
        self.n_iterations = n_iterations
        self.downscale = downscale
        self.noise_std = noise_std
        self.rotation = rotation
        self.n_perturbations = n_perturbations

    def train(self, images, group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images.

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """
        if verbose:
            print_dynamic('- Computing reference shape')
        self.reference_shape = self._compute_reference_shape(images, group,
                                                             label)
        # store number of training images
        self.n_training_images = len(images)

        # normalize the scaling of all images wrt the reference_shape size
        self._rescale_reference_shape()
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # get feature images of all levels
        images = apply_pyramid_on_images(generators, self.n_levels,
                                         verbose=verbose)

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        # build the regressors
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building regressors for each of the {} '
                              'pyramid levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building regressors\n')

        regressors = []
        # for each pyramid level (low --> high)
        for j, (level_images, level_gt_shapes) in enumerate(zip(images,
                                                                gt_shapes)):
            if verbose:
                if self.n_levels == 1:
                    print_dynamic('\n')
                elif self.n_levels > 1:
                    print_dynamic('\nLevel {}:\n'.format(j + 1))

            # build regressor
            trainer = self._set_regressor_trainer(j)
            if j == 0:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          verbose=verbose, **kwargs)
            else:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          level_shapes, verbose=verbose,
                                          **kwargs)

            if verbose:
                print_dynamic('- Perturbing shapes...')
            level_shapes = trainer.perturb_shapes(gt_shapes[0])

            regressors.append(regressor)
            count = 0
            total = len(regressors) * len(images[0]) * len(level_shapes[0])
            for k, r in enumerate(regressors):

                test_images = images[k]
                test_gt_shapes = gt_shapes[k]

                fitting_results = []
                for (i, gt_s, level_s) in zip(test_images, test_gt_shapes,
                                              level_shapes):
                    fr_list = []
                    for ls in level_s:
                        parameters = r.get_parameters(ls)
                        fr = r.fit(i, parameters)
                        fr.gt_shape = gt_s
                        fr_list.append(fr)
                        count += 1

                    fitting_results.append(fr_list)
                    if verbose:
                        print_dynamic('- Fitting shapes: {}'.format(
                            progress_bar_str((count + 1.) / total,
                                             show_bar=False)))

                level_shapes = [[Scale(self.downscale,
                                       n_dims=self.reference_shape.n_dims
                                       ).apply(fr.final_shape)
                                 for fr in fr_list]
                                for fr_list in fitting_results]

            if verbose:
                print_dynamic('- Fitting shapes: computing mean error...')
            mean_error = np.mean(np.array([fr.final_error()
                                           for fr_list in fitting_results
                                           for fr in fr_list]))
            if verbose:
                print_dynamic("- Fitting shapes: mean error "
                              "is {0:.6f}.\n".format(mean_error))

        return self._build_supervised_descent_fitter(regressors)

    def train_csy(self, images, image_path=None, lr_path=None, sample_path=None,
                  group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images. Refer to Akshay's implementation, where multiple iterations are ran.

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """

        # if given data file, then load the data manually
        # Shiyang add, for memory usage concern
        if image_path is not None:
            if len(images) is 0:
                images = []
                for (i, item) in enumerate(image_path):
                    images.extend(mio.import_pickle(item))
            else:
                raise ValueError("Should not pass both images and image_path!!")
        else:
            if len(images) is 0:
                raise ValueError("No data provided!!")

        # store number of training images
        self.n_training_images = len(images)

        if verbose:
            print_dynamic('- Computing reference shape')
        self.reference_shape = self._compute_reference_shape(images, group, label)
        # normalize the scaling of all images wrt the reference_shape size
        self._rescale_reference_shape()

        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # get feature images of all levels
        images = apply_pyramid_on_images(generators, self.n_levels,
                                         verbose=verbose)

        # Shiyang add
        # delete unused memory
        del normalized_images, generators

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        trans_models = []
        # build 2D shape models, and form transform models for later usage
        for j, level_gt_shapes in enumerate(gt_shapes):
            if verbose:
                print_dynamic('Building shape model - Level {}'.format(j))
            shape_model = build_shape_model(level_gt_shapes, None)
            shape_model.n_active_components = 16
            pdm_transform = OrthoPDM(shape_model, AlignmentSimilarity)
            trans_models.append(pdm_transform)

        # build the regressors
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building regressors for each of the {} '
                              'pyramid levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building regressors\n')

        regressors = []
        # for each pyramid level (low --> high)
        for j, (level_images, level_gt_shapes) in enumerate(zip(images,
                                                                gt_shapes)):
            if verbose:
                if self.n_levels == 1:
                    print_dynamic('\n')
                elif self.n_levels > 1:
                    print_dynamic('\nLevel {}:\n'.format(j + 1))

            # build regressor
            trainer = self._set_regressor_trainer(j)

            # perturb shape only in first iteration
            if j == 0:
                level_perturb_shapes = trainer.perturb_shapes(gt_shapes[0])

            if isinstance(self.n_iterations, list):
                max_iteration = self.n_iterations[j]
            else:
                max_iteration = self.n_iterations

            level_regressors = []
            for iter in range(max_iteration):

                regressor, estimated_delta_ps, statistics, _, _ = trainer.train_csy(
                    level_images, level_gt_shapes, level_perturb_shapes, verbose=verbose, **kwargs)

                # save the intermediate data
                if lr_path:
                    tmp = {'regressor': regressor.regressor.R, 'statistics': statistics}
                    mio.export_pickle(tmp, lr_path + "LR_lv%d_iter%d.pkl" % (j, iter), overwrite=False)
                if sample_path:
                    tmp = {'gt_shapes': level_gt_shapes, 'perturb_shapes': level_perturb_shapes,
                           'ref_shape': self.reference_shape}
                    mio.export_pickle(tmp, sample_path + "Sample_lv%d_iter%d.pkl" % (j, iter), overwrite=False)

                # update current samples
                counter = 0
                for i, p_shapes in enumerate(level_perturb_shapes):
                    for ps in p_shapes:
                        ps.points += estimated_delta_ps[counter].reshape(ps.points.shape)
                        counter += 1

                level_regressors.append(regressor)

            regressors.append(level_regressors)
            # upscale to the next level
            level_perturb_shapes = [[Scale(self.downscale, n_dims=self.reference_shape.n_dims).apply(fr)
                                     for fr in fr_list]
                                    for fr_list in level_perturb_shapes]

        return self._build_supervised_descent_fitter(regressors), trans_models

    def train_parallel(self, images, sampling_stats, trans_models, sampling_factors=None,
                       image_path=None, lr_save=False, group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images. Refer to Akshay's implementation, where multiple iterations are ran.
        And each iteration is independent.

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """

        # if given data file, then load the data manually
        # Shiyang add, for memory usage concern
        if image_path is not None:
            if len(images) is 0:
                images = []
                for (i, item) in enumerate(image_path):
                    images.extend(mio.import_pickle(item))
            else:
                raise ValueError("Should not pass both images and image_path!!")
        else:
            if len(images) is 0:
                raise ValueError("No data provided!!")

         # store number of training images
        self.n_training_images = len(images)

        # check sampling factor
        if sampling_factors:
            print_dynamic('- Use sampling factors to modify perturbation \n')

        # check sampling stats
        if sampling_stats['level'] != self.n_levels or sampling_stats['iteration'] != self.n_iterations:
            raise ValueError("Load the wrong sampling stats!")

        if sampling_stats['ref_shape']:
            self.reference_shape = sampling_stats['ref_shape']
        else:
            if verbose:
                print_dynamic('- Computing reference shape')
            self.reference_shape = self._compute_reference_shape(images, group, label)
            self._rescale_reference_shape()


        # normalize the scaling of all images wrt the reference_shape size
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # get feature images of all levels
        images = apply_pyramid_on_images(generators, self.n_levels,
                                         verbose=verbose)

        # Shiyang add
        # delete unused memory
        del normalized_images, generators

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        # build the regressors
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building regressors for each of the {} '
                              'pyramid levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building regressors\n')

        regressors = []
        LR_data = []
        # for each pyramid level (low --> high)
        for j, (level_images, level_gt_shapes) in enumerate(zip(images,
                                                                gt_shapes)):
            if verbose:
                if self.n_levels == 1:
                    print_dynamic('\n')
                elif self.n_levels > 1:
                    print_dynamic('\nLevel {}:\n'.format(j + 1))

            # build regressor
            trainer = self._set_regressor_trainer(j)

            if isinstance(self.n_iterations, list):
                max_iteration = self.n_iterations[j]
            else:
                max_iteration = self.n_iterations

            level_regressors = []
            level_LR_data = []
            for iter in range(max_iteration):

                # perturb shape in every iteration
                print_dynamic('- Sampling perturbed shapes')

                # alter the sampling range
                if j == 0 and iter == 0:
                    level_perturb_shapes = trainer.perturb_shapes(level_gt_shapes)
                else:
                    if trans_models:
                        stats = deepcopy(sampling_stats['noise_std'][j][iter])
                        # modify sampling std
                        if sampling_factors:
                            stats[:4] *= sampling_factors['global_factors'][j][iter]
                            stats[4:] *= sampling_factors['local_factors'][j][iter]

                        trans_models[j].model.n_active_components = sampling_stats['n_active_components'][j]
                        level_perturb_shapes = trainer.perturb_shapes_pdm(level_gt_shapes,
                            trans_models[j], stats)
                    else:
                        level_perturb_shapes = trainer.perturb_shapes_direct(
                            level_gt_shapes, sampling_stats['noise_std'][j][iter])

                regressor, _, _, features, delta_ps = trainer.train_csy(
                    level_images, level_gt_shapes, level_perturb_shapes, verbose=verbose, **kwargs)

                if lr_save:
                    # # calculate model terms for incremental SDM (use eigen analysis to accelerate)
                    # features = np.hstack((features, np.ones(features.shape[0])[:, None]))
                    # delta_ps = np.hstack((delta_ps, np.ones(delta_ps.shape[0])[:, None]))
                    # u, s, v = np.linalg.svd(features)
                    # tol = np.max(features.shape) * np.spacing(np.max(s))
                    # sel = np.sum(s > tol)
                    # s = np.diag(1.0 / s[:sel])
                    # LR_V = v[:, :sel].dot(s**2).dot(v[:, :sel].T)
                    # LR_W = v[:, :sel].dot(s).dot(u[:, :sel].T).dot(delta_ps)

                    XX = np.dot(features.T, features)
                    LR_V = np.linalg.pinv((XX + XX.T)/2)
                    tmp = {'V': LR_V, 'W': regressor.regressor.R}
                    level_LR_data.append(tmp)

                level_regressors.append(regressor)

            LR_data.append(level_LR_data)
            regressors.append(level_regressors)

        return self._build_supervised_descent_fitter(regressors), LR_data

    def train_iPar(self, images, sampling_stats, trans_models, lr_data_path, multi_step_rls,
                   sampling_factors=None, lr_save_path=None, image_path=None,
                   group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images. Incremental formulation for parallel sampling

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """

        # if given data file, then load the data manually
        # Shiyang add, for memory usage concern
        if image_path is not None:
            if len(images) is 0:
                images = []
                for (i, item) in enumerate(image_path):
                    images.extend(mio.import_pickle(item))
            else:
                raise ValueError("Should not pass both images and image_path!!")
        else:
            if len(images) is 0:
                raise ValueError("No data provided!!")

        # store number of training images
        self.n_training_images = len(images)

        # check sampling stats
        if sampling_stats['level'] != self.n_levels or sampling_stats['iteration'] != self.n_iterations:
            raise ValueError("Load the wrong sampling stats!")

        # check sampling factor
        if sampling_factors:
            print_dynamic('- Use sampling factors to modify perturbation \n')

        if sampling_stats['ref_shape']:
            self.reference_shape = sampling_stats['ref_shape']
        else:
            if verbose:
                print_dynamic('- Computing reference shape')
            self.reference_shape = self._compute_reference_shape(images, group, label)
            self._rescale_reference_shape()


        # normalize the scaling of all images wrt the reference_shape size
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # get feature images of all levels
        images = apply_pyramid_on_images(generators, self.n_levels,
                                         verbose=verbose)

        # Shiyang add
        # delete unused memory
        del normalized_images, generators

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        # load initial LR models
        init_lr_models = mio.import_pickle(lr_data_path)
        # prepare a set of trainers and regressors
        trainers = []
        init_regressors = []
        for i, models in enumerate(init_lr_models):
            level_regressors = []
            trainers.append(self._set_regressor_trainer(i))
            for j, m in enumerate(models):
                tmp = trainers[i].regression_type(m['W'], m['V'], multi_step=multi_step_rls)
                level_regressors.append(trainers[i]._build_regressor(tmp, trainers[i].features))
            init_regressors.append(level_regressors)

        # start incrementally update the models
        for image_idx in range(self.n_training_images):
            if verbose:
                print_dynamic('Processing image - {}'.format(progress_bar_str(
                    (image_idx + 1.)/self.n_training_images, show_bar=True)))

            # for each pyramid level (low --> high)
            for j, (level_images, level_gt_shapes) in enumerate(zip(images, gt_shapes)):
                level_gt_shapes = [level_gt_shapes[image_idx]]
                level_images = [level_images[image_idx]]

                if isinstance(self.n_iterations, list):
                    max_iteration = self.n_iterations[j]
                else:
                    max_iteration = self.n_iterations

                for iter in range(max_iteration):
                    # alter the sampling range
                    if j == 0 and iter == 0:
                        level_perturb_shapes = trainers[j].perturb_shapes(level_gt_shapes)
                    else:
                        trans_models[j].model.n_active_components = sampling_stats['n_active_components'][j]
                        stats = deepcopy(sampling_stats['noise_std'][j][iter])
                        # modify sampling std
                        if sampling_factors:
                            stats[:4] *= sampling_factors['global_factors'][j][iter]
                            stats[4:] *= sampling_factors['local_factors'][j][iter]

                        level_perturb_shapes = trainers[j].perturb_shapes_pdm(level_gt_shapes,
                            trans_models[j], stats)

                    features, delta_ps = trainers[j].compute_feature(
                        level_images, level_gt_shapes, level_perturb_shapes, verbose=False, **kwargs)

                    # update regressor
                    init_regressors[j][iter].regressor.update_model(features, delta_ps)

            if lr_save_path:
                if (image_idx+1) % 500 == 0:
                    sdm = self._build_supervised_descent_fitter(init_regressors)
                    if multi_step_rls:
                        final_path = lr_save_path + '_multi_{}.pkl'.format(image_idx+1)
                    else:
                        final_path = lr_save_path + '_single_{}.pkl'.format(image_idx+1)
                    with open(final_path, 'wb') as f:
                        dill.dump(sdm, f)

        # save LR data for further update
        LR_data = []
        for i, regs in enumerate(init_regressors):
            level_LR_data = []
            for j, r in enumerate(regs):
                tmp = {'V': r.regressor.V, 'W': r.regressor.R}
                level_LR_data.append(tmp)
            LR_data.append(level_LR_data)

        return self._build_supervised_descent_fitter(init_regressors), LR_data

    def train_iPar_tracking(self, images, fittings, sampling_stats, trans_models, LR_data, init_regressors,
                   sampling_factors=None, group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images. Incremental formulation for parallel sampling

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """

        if len(images) is 0 and init_regressors:
            raise ValueError("No data provided!!")

        # attach landmarks
        if len(images) > 0 and len(fittings) > 0:
            for i, (im, p) in enumerate(zip(images, fittings)):
                # attach landmarks to the image
                im.landmarks[group] = p

        # store number of training images
        self.n_training_images = len(images)

        # check sampling stats
        if sampling_stats['level'] != self.n_levels or sampling_stats['iteration'] != self.n_iterations:
            raise ValueError("Load the wrong sampling stats!")

        # check sampling factor
        if sampling_factors and verbose:
            print_dynamic('- Use sampling factors to modify perturbation \n')


        self.reference_shape = sampling_stats['ref_shape']

        # normalize the scaling of all images wrt the reference_shape size
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # get feature images of all levels
        images = apply_pyramid_on_images(generators, self.n_levels,
                                         verbose=verbose)

        # Shiyang add
        # delete unused memory
        del normalized_images, generators

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        # prepare a set of trainers and regressors
        if init_regressors:
            trainers = []
            for i, models in enumerate(init_regressors):
                trainers.append(self._set_regressor_trainer(i))
        else:
            trainers = []
            init_regressors = []
            for i, models in enumerate(LR_data):
                level_regressors = []
                trainers.append(self._set_regressor_trainer(i))
                for j, m in enumerate(models):
                    tmp = trainers[i].regression_type(m['W'], m['V'], multi_step=True)
                    level_regressors.append(trainers[i]._build_regressor(tmp, trainers[i].features))
                init_regressors.append(level_regressors)

        # start incrementally update the models
        for image_idx in range(self.n_training_images):
            # for each pyramid level (low --> high)
            for j, (level_images, level_gt_shapes) in enumerate(zip(images, gt_shapes)):
                level_gt_shapes = [level_gt_shapes[image_idx]]
                level_images = [level_images[image_idx]]

                if isinstance(self.n_iterations, list):
                    max_iteration = self.n_iterations[j]
                else:
                    max_iteration = self.n_iterations

                for iter in range(max_iteration):
                    # alter the sampling range
                    if j == 0 and iter == 0:
                        level_perturb_shapes = trainers[j].perturb_shapes(level_gt_shapes)
                    else:
                        trans_models[j].model.n_active_components = sampling_stats['n_active_components'][j]
                        stats = deepcopy(sampling_stats['noise_std'][j][iter])
                        # modify sampling std
                        if sampling_factors:
                            stats[:4] *= sampling_factors['global_factors'][j][iter]
                            stats[4:] *= sampling_factors['local_factors'][j][iter]

                        level_perturb_shapes = trainers[j].perturb_shapes_pdm(level_gt_shapes,
                            trans_models[j], stats)

                    features, delta_ps = trainers[j].compute_feature(
                        level_images, level_gt_shapes, level_perturb_shapes, verbose=False, **kwargs)

                    # update regressor
                    init_regressors[j][iter].regressor.update_model(features, delta_ps)

        return self._build_supervised_descent_fitter(init_regressors)

    @classmethod
    def _normalization_wrt_reference_shape(cls, images, group, label,
                                           reference_shape, verbose=False):
        r"""
        Normalizes the images sizes with respect to the reference
        shape (mean shape) scaling. This step is essential before building a
        deformable model.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the model.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        reference_shape : :map:`PointCloud`
            The reference shape that is used to resize all training images to
            a consistent object size.

        verbose: bool, optional
            Flag that controls information and progress printing.

        Returns
        -------
        normalized_images : :map:`MaskedImage` list
            A list with the normalized images.
        """
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images),
                                     show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                reference_shape, group=group, label=label))

        if verbose:
            print_dynamic('- Normalizing images size: Done\n')
        return normalized_images

    @abc.abstractmethod
    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on the given images shapes.
        """
        pass

    def _rescale_reference_shape(self):
        r"""
        Function that rescales the reference shape w.r.t. to
        ``normalization_diagonal`` parameter.
        """
        pass

    @abc.abstractmethod
    def _set_regressor_trainer(self, **kwargs):
        r"""
        Function that sets the regression object to be one from
        :map:`RegressorTrainer`,
        """
        pass

    @abc.abstractmethod
    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object.

        Parameters
        ----------
        regressors : list of :map:`RegressorTrainer`
            The list of regressors.

        Returns
        -------
        fitter : :map:`SDMFitter`
            The SDM fitter object.
        """
        pass


class SDMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Method using Non-Parametric
    Regression.

    Parameters
    ----------
    regression_type : `callable` or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        The callable should be one of the methods defined in
        :ref:`regression_callables`

    regression_features: ``None`` or  `callable` or `[callable]`, optional
        If list of length ``n_levels``, then a feature is defined per level.

        If not a list, then the specified feature will be applied to all
        pyramid levels.

        Per level:
            If ``None``, no features are extracted, thus specified
            ``features`` is used in the regressor.

            It is recommended to set the desired features using this option,
            leaving ``features`` equal to :map:`no_op`. This means that the
            images will remain in the intensities space and the features will
            be extracted by the regressor.

    patch_shape: tuple of `int`
        The shape of the patches used by the SDM.

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        initial shape.

    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the initial shape.

    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the shapes.

    normalization_diagonal : `int` >= ``20``, optional
        During training, all images are rescaled to ensure that the scale of
        their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    Raises
    ------
    ValueError
        ``regression_features`` must be ``None`` or a `string` or a `function`
        or a list of those containing 1 or ``n_level`` elements
    """
    def __init__(self, regression_type=mlr, regression_features=sparse_hog,
                 patch_shape=(16, 16), features=no_op, n_levels=3, n_iterations=10,
                 downscale=1.5, noise_std=0.04,
                 rotation=False, n_perturbations=10,
                 normalization_diagonal=None):
        super(SDMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            features=features, n_levels=n_levels, n_iterations=n_iterations,
            downscale=downscale, noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations)
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on the given images.
        """
        shapes = [i.landmarks[group][label] for i in images]
        return mean_pointcloud(shapes)

    def _rescale_reference_shape(self):
        r"""
        Function that rescales the reference shape w.r.t. to
        ``normalization_diagonal`` parameter.
        """
        if self.normalization_diagonal:
            x, y = self.reference_shape.range()
            scale = self.normalization_diagonal / np.sqrt(x**2 + y**2)
            Scale(scale, self.reference_shape.n_dims).apply_inplace(
                self.reference_shape)

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`NonParametricRegressorTrainer`.

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer : :map:`NonParametricRegressorTrainer`
            The regressor object.
        """
        return NonParametricRegressorTrainer(
            self.reference_shape, regression_type=self.regression_type[level],
            regression_features=self.regression_features[level],
            patch_shape=self.patch_shape, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object.

        Parameters
        ----------
        regressors : list of :map:`RegressorTrainer`
            The list of regressors.

        Returns
        -------
        fitter : :map:`SDMFitter`
            The SDM fitter object.
        """
        return SDMFitter(regressors, self.n_training_images, self.features,
                         self.reference_shape, self.downscale)


class SDAAMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Regressor for a given Active
    Appearance Model, thus uses Parametric Regression.

    Parameters
    ----------
    aam : :map:`AAM`
        The trained AAM object.
    regression_type : `callable`, or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        Examples of such callables can be found in :ref:`regression_callables`.
    regression_features: `function` or list of those, optional
        If list of length ``n_levels``, then a feature is defined per level.

        If not a list or a list with length ``1``, then the specified feature
        will be applied to all pyramid levels.

        The callable should be one of the methods defined in
        :ref:`parametricfeatures`.
    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.
    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.
    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.
    update : {'additive', 'compositional'}
        Defines the way that the warp will be updated.
    md_transform: :map:`ModelDrivenTransform`, optional
        The model driven transform class to be used.
    n_shape : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of shape components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available shape components
            (``n_active_components``)will be used.

            If `int` > ``1``, a specific number of shape components is
            specified.

            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.
    n_appearance : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of appearance components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available appearance components
            (``n_active_components``) will be used.
            
            If `int > ``1``, a specific number of appearance components is
            specified.
            
            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.

    Raises
    -------
    ValueError
        n_shape can be an integer or a float or None or a list containing 1
        or ``n_levels`` of those
    ValueError
        n_appearance can be an integer or a float or None or a list containing
        1 or ``n_levels`` of those
    ValueError
        ``regression_features`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    def __init__(self, aam, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                 update='compositional', md_transform=OrthoMDTransform,
                 n_shape=None, n_appearance=None):
        super(SDAAMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            features=aam.features, n_levels=aam.n_levels,
            downscale=aam.downscale, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.aam = aam
        self.update = update
        self.md_transform = md_transform
        # hard coded for now as this is the only supported configuration.
        self.global_transform = DifferentiableAlignmentSimilarity

        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.aam.n_levels:
                for sm, n in zip(self.aam.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.aam.n_levels))

        # check n_appearance parameter
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.aam.n_levels))

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that returns the reference shape computed during AAM building.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on.
        """
        return self.aam.reference_shape

    def _normalize_object_size(self, images, group, label):
        r"""
        Function that normalizes the images sizes with respect to the reference
        shape (mean shape) scaling.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the model.

        group : `string`
            The key of the landmark set that should be used. If ```None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        normalized_images : :map:`MaskedImage` list
            A list with the normalized images.
        """
        return [i.rescale_to_reference_shape(self.reference_shape,
                                             group=group, label=label)
                for i in images]

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`ParametricRegressorTrainer`.

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer: :map:`ParametricRegressorTrainer`
            The regressor object.
        """
        am = self.aam.appearance_models[level]
        sm = self.aam.shape_models[level]

        if self.md_transform is not ModelDrivenTransform:
            md_transform = self.md_transform(
                sm, self.aam.transform, self.global_transform,
                source=am.mean().landmarks['source'].lms)
        else:
            md_transform = self.md_transform(
                sm, self.aam.transform,
                source=am.mean().landmarks['source'].lms)

        return ParametricRegressorTrainer(
            am, md_transform, self.reference_shape,
            regression_type=self.regression_type[level],
            regression_features=self.regression_features[level],
            update=self.update, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object for AAMs.

        Parameters
        ----------
        regressors : :map:`RegressorTrainer`
            The regressor to build with.

        Returns
        -------
        fitter : :map:`SDAAMFitter`
            The SDM fitter object.
        """
        return SDAAMFitter(self.aam, regressors, self.n_training_images)


class SDCLMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Regressor for a given Constrained
    Local Model, thus uses Semi Parametric Classifier-Based Regression.

    Parameters
    ----------
    clm : :map:`CLM`
        The trained CLM object.
    regression_type : `callable`, or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        Examples of such callables can be found in :ref:`regression_callables`.
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.
    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.
    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.
    pdm_transform : :map:`ModelDrivenTransform`, optional
        The point distribution transform class to be used.
    n_shape : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of shape components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available shape components
            (``n_active_components``) will be used.

            If `int` > ``1``, a specific number of shape components is
            specified.
            
            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.

    Raises
    -------
    ValueError
        ``n_shape`` can be an integer or a `float` or ``None`` or a list
        containing ``1`` or ``n_levels`` of those.
    """
    def __init__(self, clm, regression_type=mlr, noise_std=0.04,
                 rotation=False, n_perturbations=10, pdm_transform=OrthoPDM,
                 n_shape=None):
        super(SDCLMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=[None] * clm.n_levels,
            features=clm.features, n_levels=clm.n_levels,
            downscale=clm.downscale, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.clm = clm
        self.patch_shape = clm.patch_shape
        self.pdm_transform = pdm_transform
        # hard coded for now as this is the only supported configuration.
        self.global_transform = DifferentiableAlignmentSimilarity

        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.clm.n_levels > 1:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.clm.n_levels:
                for sm, n in zip(self.clm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.clm.n_levels))

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that returns the reference shape computed during CLM building.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape.
        """
        return self.clm.reference_shape

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`SemiParametricClassifierBasedRegressorTrainer`

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer: :map:`SemiParametricClassifierBasedRegressorTrainer`
            The regressor object.
        """
        clfs = self.clm.classifiers[level]
        sm = self.clm.shape_models[level]

        if self.pdm_transform is not PDM:
            pdm_transform = self.pdm_transform(sm, self.global_transform)
        else:
            pdm_transform = self.pdm_transform(sm)

        return SemiParametricClassifierBasedRegressorTrainer(
            clfs, pdm_transform, self.reference_shape,
            regression_type=self.regression_type[level],
            patch_shape=self.patch_shape, update='additive',
            noise_std=self.noise_std, rotation=self.rotation,
            n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object for CLMs.

        Parameters
        ----------
        regressors : :map:`RegressorTrainer`
            Regressor to train with.

        Returns
        -------
        fitter : :map:`SDCLMFitter`
            The SDM fitter object.
        """
        return SDCLMFitter(self.clm, regressors, self.n_training_images)
