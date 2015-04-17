from __future__ import division
import abc
import numpy as np
from menpo.image import Image
from menpo.feature import gradient as fast_gradient
from .result import AAMAlgorithmResult


class AAMInterface(object):

    def __init__(self, aam_algorithm):
        self.algorithm = aam_algorithm

        # grab algorithm transform
        self.transform = self.algorithm.transform
        # grab number of shape parameters
        self.n = self.transform.n_parameters
        # grab algorithm appearance model
        self.appearance_model = self.algorithm.appearance_model
        # grab number of appearance parameters
        self.m = self.appearance_model.n_active_components

    @abc.abstractmethod
    def warp_jacobian(self):
        pass

    @abc.abstractmethod
    def warp(self, image):
        pass

    @abc.abstractmethod
    def gradient(self, image):
        pass

    @abc.abstractmethod
    def steepest_descent_images(self, nabla, dW_dp):
        pass

    @classmethod
    def solve_shape_map(cls, H, J, e, J_prior, p):
        # compute and return MAP solution
        H += np.diag(J_prior)
        Je = J_prior * p + J.T.dot(e)
        return -np.linalg.solve(H, Je)

    @classmethod
    def solve_shape_ml(cls, H, J, e):
        # compute and return ML solution
        return -np.linalg.solve(H, J.T.dot(e))

    @abc.abstractmethod
    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        pass


class GlobalAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(GlobalAAMInterface, self). __init__(aam_algorithm)

        # grab algorithm shape model
        self.shape_model = self.transform.pdm.model
        # grab algorithm template
        self.template = self.algorithm.template
        # grab algorithm template mask true indices
        self.true_indices = self.template.mask.true_indices()

        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters
        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling_step is None:
            sampling_step = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

    def warp_jacobian(self):
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.algorithm.template.mask,
                                  self.algorithm.transform)

    def gradient(self, image):
        nabla = fast_gradient(image)
        nabla.set_boundary_pixels()
        return nabla.as_vector().reshape((2, image.n_channels, -1))

    def steepest_descent_images(self, nabla, dW_dp):
        # reshape gradient
        # nabla: n_dims x n_channels x n_pixels
        nabla = nabla[self.nabla_mask].reshape(nabla.shape[:2] + (-1,))
        # compute steepest descent images
        # nabla: n_dims x n_channels x n_pixels
        # warp_jacobian: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d
        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2])).dot(self.transform.Jp().T)

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        self.norm_func = self.appearance_model.norm_func
        # grab algorithm shape model
        self.shape_model = self.transform.model
        # grab appearance model parts shape
        self.parts_shape = self.appearance_model.parts_shape

        if sampling_mask is None:
            sampling_mask = np.ones(self.parts_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.nabla_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_size=self.parts_shape,
                                      as_single_array=True)
        if self.norm_func:
            parts = self.norm_func(parts)
        return Image(parts)

    def gradient(self, image):
        nabla = fast_gradient(image.pixels.reshape((-1,) + self.parts_shape))
        return nabla.reshape((2,) + image.pixels.shape)

    def steepest_descent_images(self, nabla, dW_dp):
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.nabla_mask].reshape(
            nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * dW_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AAMAlgorithm(object):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.eps = eps
        # set interface
        self.interface = aam_interface(self, **kwargs)
        # perform pre-computations
        self.precompute()

    def precompute(self):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters
        self.m = self.appearance_model.n_active_components

        # grab appearance model components
        self.A = self.appearance_model.components
        # mask them
        self.A_m = self.A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        s2 = (self.appearance_model.noise_variance() /
              self.interface.shape_model.variance())
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))
        # compute appearance model prior
        S = self.appearance_model.eigenvalues
        self.s2_inv_S = s2 / S

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        pass


class ProjectOut(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def project_out(self, J):
        r"""
        Project-out appearance bases from a particular vector or matrix
        """
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Project-out AAM algorithms
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # solve for increments on the shape parameters
            self.dp = self.solve(map_inference)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def solve(self, map_inference):
        pass

    @abc.abstractmethod
    def update_warp(self):
        r"""
        Update warp
        """
        pass


class PIC(ProjectOut):
    r"""
    Project-out Inverse Compositional Gauss-Newton algorithm
    """
    def precompute(self):
        r"""
        Pre-compute PIC state
        """
        # call super method
        super(PIC, self).precompute()

        # compute appearance model mean gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # compute masked inverse Jacobian
        J_m = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # project out appearance model from it
        self.QJ_m = self.project_out(J_m)
        # compute masked Hessian
        self.JQJ_m = self.QJ_m.T.dot(J_m)
        # compute masked Jacobian pseudo-inverse
        self.pinv_QJ_m = np.linalg.solve(self.JQJ_m, self.QJ_m.T)

    def solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JQJ_m, self.QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_QJ_m.dot(self.e_m)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class Alternating(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

    def precompute(self, **kwargs):
        r"""
        Pre-compute common state for Alternating algorithms
        """
        # call super method
        super(Alternating, self).precompute()

        self.AA_m_map = self.A_m.T.dot(self.A_m) + np.diag(self.s2_inv_S)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Alternating AAM algorithms
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [c]
                Jdp = 0
            else:
                Jdp = J_m.dot(self.dp)

            # compute masked error
            e_m = i_m - a_m

            # solve for increment on the appearance parameters
            if map_inference:
                Ae_m_map = - self.s2_inv_S * c + self.A_m.T.dot(e_m + Jdp)
                dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
            else:
                dc = self.pinv_A_m.dot(e_m + Jdp)

            # compute masked  Jacobian
            J_m = self.compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m - self.A_m.dot(dc), self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m,
                                                        e_m - self.A_m.dot(dc))

            # update appearance parameters
            c += dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def compute_jacobian(self):
        r"""
        Compute Jacobian
        """
        pass

    @abc.abstractmethod
    def update_warp(self):
        r"""
        Update warp
        """
        pass


class AIC(Alternating):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Inverse Jacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)