from __future__ import division
from menpofit.base import noisy_align
from menpofit.fitter import align_shape_with_bb
from alaborticcv2015.alignment.fitter import Fitter
from alaborticcv2015.alignment.pdm import OrthoPDM
from alaborticcv2015.alignment.transform import OrthoMDTransform
from .algorithm import GlobalAAMInterface, PartsAAMInterface, AIC


class AAMFitter(Fitter):

    @property
    def reference_shape(self):
        r"""
        The reference shape of the AAM.

        :type: :map:`PointCloud`
        """
        return self.dm.reference_shape

    @property
    def features(self):
        r"""
        The feature extracted at each pyramidal level during AAM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self.dm.features

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during AAM building.

        :type: `int`
        """
        return self.dm.n_levels

    @property
    def scales(self):
        return self.dm.scales

    @property
    def scale_features(self):
        r"""
        Flag that defined the nature of Gaussian pyramid used to build the
        AAM.
        If ``True``, the feature space is computed once at the highest scale
        and the Gaussian pyramid is applied to the feature images.
        If ``False``, the Gaussian pyramid is applied to the original images
        and features are extracted at each level.

        :type: `boolean`
        """
        return self.dm.scale_features

    def _check_n_shape(self, n_shape):
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.dm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.dm.n_levels > 1:
                for sm in self.dm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.dm.n_levels:
                for sm, n in zip(self.dm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.dm.n_levels > 1:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.dm.n_levels:
                for am, n in zip(self.dm.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    def perturb_shape(self, gt_shape, noise_std=0.04, rotation=False):
        r"""
        Generates an initial shape by adding gaussian noise to the perfect
        similarity alignment between the ground truth and reference_shape.

        Parameters
        -----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        noise_std: float, optional
            The standard deviation of the gaussian noise used to produce the
            initial shape.

            Default: 0.04
        rotation: boolean, optional
            Specifies whether ground truth in-plane rotation is to be used
            to produce the initial shape.

            Default: False

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """
        reference_shape = self.reference_shape
        return noisy_align(reference_shape, gt_shape, noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def obtain_shape_from_bb(self, bounding_box):
        r"""
        Generates an initial shape given a bounding box detection.

        Parameters
        -----------
        bounding_box: (2, 2) ndarray
            The bounding box specified as:

                np.array([[x_min, y_min], [x_max, y_max]])

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """

        reference_shape = self.reference_shape
        return align_shape_with_bb(reference_shape,
                                   bounding_box).apply(reference_shape)


class GlobalAAMFitter(AAMFitter):

    def __init__(self, global_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):
        self.dm = global_aam
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=am.mean().landmarks['source'].lms,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(GlobalAAMInterface, am,
                                      md_transform, **kwargs)

            self.algorithms.append(algorithm)


class PartsAAMFitter(AAMFitter):

    def __init__(self, parts_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):
        self.dm = parts_aam
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            pdm = OrthoPDM(sm, sigma2=am.noise_variance())

            am.parts_shape = self.dm.parts_shape
            am.norm_func = self.dm.norm_func
            algorithm = algorithm_cls(PartsAAMInterface, am, pdm, **kwargs)

            self.algorithms.append(algorithm)
