import numpy as np
from menpo.image import Image

from menpofit.base import DeformableModel

# Shiyang add
import menpo.io as mio
from menpo.visualize import print_dynamic, progress_bar_str
from menpofit.base import create_pyramid, build_sampling_grid
from .builder import get_pos_neg_grid_positions, extract_scroll_window


class CLM(DeformableModel):
    r"""
    Constrained Local Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the CLM.

    classifiers : ``[[callable]]``
        A list containing the list of classifier callables per each pyramidal
        level of the CLM.

    n_training_images : `int`
        The number of training images used to build the AAM.

    patch_shape : tuple of `int`
        The shape of the patches used to train the classifiers.

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, Optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

    """
    def __init__(self, shape_models, classifiers, n_training_images,
                 patch_shape, patch_size, features, reference_shape, downscale,
                 scaled_shape_models, use_scroll_window):
        DeformableModel.__init__(self, features)
        self.shape_models = shape_models
        self.classifiers = classifiers
        self.n_training_images = n_training_images
        self.patch_shape = patch_shape
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models

        # Shiyang add
        self.patch_size = patch_size
        self.use_scroll_window = use_scroll_window

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the CLM.

        :type: `int`
        """
        return len(self.shape_models)

    @property
    def n_classifiers_per_level(self):
        """
        The number of classifiers per pyramidal level of the CLM.

        :type: `int`
        """
        return [len(clf) for clf in self.classifiers]

    def instance(self, shape_weights=None, level=-1):
        r"""
        Generates a novel CLM instance given a set of shape weights. If no
        weights are provided, the mean CLM instance is returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If `None`, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        shape_instance : :map:`PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def random_instance(self, level=-1):
        r"""
        Generates a novel random CLM instance.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        shape_instance : :map:`PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def response_image(self, image, group=None, label=None, level=-1):
        r"""
        Generates a response image result of applying the classifiers of a
        particular pyramidal level of the CLM to an image.

        Parameters
        -----------
        image: :map:`Image`
            The image.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label : `string`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        level: `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The response image.
        """
        # rescale image
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group=group, label=label)

        # apply pyramid
        if self.n_levels > 1:
            if self.pyramid_on_features:
                # compute features at highest level
                feature_image = self.features(image)

                # apply pyramid on feature image
                pyramid = feature_image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # get rescaled feature images
                images = list(pyramid)
            else:
                # create pyramid on intensities image
                pyramid = image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # compute features at each level
                images = [self.features[self.n_levels - j - 1](i)
                          for j, i in enumerate(pyramid)]
            images.reverse()
        else:
            images = [self.features(image)]

        # initialize responses
        image = images[level]
        image_pixels = np.reshape(image.pixels, (-1, image.n_channels))
        response_data = np.zeros((image.shape[0], image.shape[1],
                                  self.n_classifiers_per_level[level]))
        # Compute responses
        for j, clf in enumerate(self.classifiers[level]):
            response_data[:, :, j] = np.reshape(clf(image_pixels),
                                                image.shape)
        return Image(image_data=response_data)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        : str
        """
        return 'Constrained Local Model'

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0), mode='multiple',
                                 popup=False, figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the CLM object using the
        `menpo.visualize.widgets.visualize_shape_model` widget.

        Parameters
        -----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders.
            If `int`, then the number of sliders per level is the minimum
            between `n_parameters` and the number of active components per
            level.
            If `list` of `int`, then a number of sliders is defined per level.
            If ``None``, all the active components per level will have a slider.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu.
            If ``'multiple'``, a slider is constructed for each parameter.
        popup : `bool`, optional
            If ``True``, the widget will appear as a popup window.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        """
        from menpofit.visualize import visualize_shape_model
        visualize_shape_model(self.shape_models, n_parameters=n_parameters,
                              parameters_bounds=parameters_bounds,
                              figure_size=figure_size, mode=mode, popup=popup)

    def __str__(self):
        from menpofit.base import name_of_callable
        out = "{}\n - {} training images.\n".format(self._str_title,
                                                    self.n_training_images)
        # small strings about number of channels, channels string and downscale
        down_str = []
        for j in range(self.n_levels):
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        temp_img = Image(image_data=np.random.rand(50, 50))
        if self.pyramid_on_features:
            temp = self.features(temp_img)
            n_channels = [temp.n_channels] * self.n_levels
        else:
            n_channels = []
            for j in range(self.n_levels):
                temp = self.features[j](temp_img)
                n_channels.append(temp.n_channels)
        # string about features and channels
        if self.pyramid_on_features:
            feat_str = "- Feature is {} with ".format(
                name_of_callable(self.features))
            if n_channels[0] == 1:
                ch_str = ["channel"]
            else:
                ch_str = ["channels"]
        else:
            feat_str = []
            ch_str = []
            for j in range(self.n_levels):
                feat_str.append("- Feature is {} with ".format(
                    name_of_callable(self.features[j])))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        if self.n_levels > 1:
            if self.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.patch_shape[1], self.patch_shape[0])

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.patch_shape[1], self.patch_shape[0])
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                out = "{0}     - {1} shape components ({2:.2f}% of " \
                      "variance)\n     - {3} {4} classifiers.\n".format(
                    out, self.shape_models[i].n_components,
                    self.shape_models[i].variance_ratio() * 100,
                    self.n_classifiers_per_level[i],
                    name_of_callable(self.classifiers[i][0]))
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - {4} shape components ({5:.2f}% of " \
                  "variance)\n   - {6} {7} classifiers.".format(
                out, feat_str[0], n_channels[0], ch_str[0],
                self.shape_models[0].n_components,
                self.shape_models[0].variance_ratio() * 100,
                self.n_classifiers_per_level[0],
                name_of_callable(self.classifiers[0][0]))
        return out

    def update_patch_expert(self, images, image_path=None,
                            group=None, label=None, verbose=False):
        """
        extract the new training samples, and update the patch experts
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

        print_dynamic('Start to update the patch experts.\n')

        # normalize the scaling of all images wrt the reference_shape size
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images), show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                self.reference_shape, group=group, label=label))

        if verbose:
            print_dynamic('- Normalizing images size: Done\n')

        # Shiyang add
        if image_path is not None:
            del images

        generators = create_pyramid(normalized_images, self.n_levels, self.downscale,
                                    self.features, verbose=verbose)
        # Shiyang add
        self.n_training_images += len(normalized_images)
        del normalized_images

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Updating model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Updating model\n')

        # # start to update the patch experts
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
                            level_str, progress_bar_str((c + 1.) / len(generators), show_bar=False)))
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # build classifiers
            sampling_grid = build_sampling_grid(self.patch_shape)

            # Shiyang add
            if self.n_classifiers_per_level[rj] != shapes[0].n_points:
                raise ValueError("New data doesn't have same # of points!")
            else:
                n_points = self.n_classifiers_per_level[rj]

            for k in range(n_points):
                if verbose:
                    print_dynamic('{}Updating classifiers - {}'.format(
                        level_str, progress_bar_str((k + 1.) / n_points, show_bar=False)))

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

                # update classifier
                self.classifiers[rj][k].update(X, t)

                # tmp = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\tmp\num_" +
                #                         repr(self.n_training_images-len(feature_images)) +
                #                         r"\pn_" + repr(j) + '_' + repr(k) + ".pkl")
                # self.classifiers[rj][k].update(X, t, tmp[0], tmp[1])

                # if self.n_training_images < 6000:
                #     mio.export_pickle([np.vstack((X, tmp[0])), np.vstack((t[:, None], tmp[1][:, None]))[:, 0]],
                #                       r"C:\Csy\incremental-alignment\CLM\tmp\num_" + repr(self.n_training_images) +
                #                       r"\pn_" + repr(j) + '_' + repr(k) + ".pkl", overwrite=True)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

    def update_patch_expert_1by1(self, images, image_path=None,
                            group=None, label=None, verbose=False):
        """
        extract the new training samples, and update the patch experts, with one image at a time
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

        print_dynamic('Start to update the patch experts.\n')

        # normalize the scaling of all images wrt the reference_shape size
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images), show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                self.reference_shape, group=group, label=label))

        if verbose:
            print_dynamic('- Normalizing images size: Done\n')

        # Shiyang add
        if image_path is not None:
            del images

        generators = create_pyramid(normalized_images, self.n_levels, self.downscale,
                                    self.features, verbose=verbose)
        # Shiyang add
        self.n_training_images += len(normalized_images)
        del normalized_images

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Updating model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Updating model\n')

        # # start to update the patch experts
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
                            level_str, progress_bar_str((c + 1.) / len(generators), show_bar=False)))
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # build classifiers
            sampling_grid = build_sampling_grid(self.patch_shape)

            # Shiyang add
            if self.n_classifiers_per_level[rj] != shapes[0].n_points:
                raise ValueError("New data doesn't have same # of points!")
            else:
                n_points = self.n_classifiers_per_level[rj]

            for k in range(n_points):
                if verbose:
                    print_dynamic('{}Updating classifiers - {}'.format(
                        level_str, progress_bar_str((k + 1.) / n_points, show_bar=False)))

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

                    # Shiyang change here
                    positive_samples = np.asanyarray(positive_sample)
                    positive_samples = np.reshape(positive_samples,
                                                  (-1, positive_samples.shape[-1]))
                    positive_labels = np.asanyarray(np.ones(positive_sample.shape[0])).flatten()

                    negative_samples = np.asanyarray(negative_sample)
                    negative_samples = np.reshape(negative_samples,
                                                  (-1, negative_samples.shape[-1]))
                    negative_labels = np.asanyarray(-np.ones(negative_sample.shape[0])).flatten()

                    X = np.vstack((positive_samples, negative_samples))
                    t = np.hstack((positive_labels, negative_labels))

                    # update classifier
                    self.classifiers[rj][k].max_iter_ = 20
                    self.classifiers[rj][k].lambda_ = 0.001
                    self.classifiers[rj][k].update(X, t)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))