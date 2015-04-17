from __future__ import division
from alaborticcv2015.alignment.result import AlgorithmResult


class LKAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters, gt_shape=None):
        super(LKAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self._gt_shape = gt_shape
