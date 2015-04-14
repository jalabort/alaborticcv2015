from menpo.feature import centralize
from .base import LinDeepConvNet, _parse_filters, normalize_filters


class PreTrainedLDCN(LinDeepConvNet):
    r"""
    Pre-trained Linear Deep Convolutional Network Class
    """
    def __init__(self, architecture=3, norm_func=centralize):
        super(PreTrainedLDCN, self).__init__(architecture=architecture)
        self.norm_func = norm_func

    def build_network(self, filters):
        self._filters = normalize_filters(filters, self.norm_func)
        self._n_filters = _parse_filters(self._filters)


