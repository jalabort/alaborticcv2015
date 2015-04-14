from alaborticcv2015.utils import centralize, normalize_patches
from .base import LinDeepConvNet, _parse_filters


class PreTrainedLDCN(LinDeepConvNet):

    def __init__(self, norm_func=centralize):
        self.norm_func = norm_func

    def build_network(self, filters):
        self._filters = filters
        self._n_filters = _parse_filters(self._filters)
        self._normalize_filters()

