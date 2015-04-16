from .base import LinDeepConvNet, _parse_filters, _normalize_filters


class PreTrainedLDCN(LinDeepConvNet):
    r"""
    Pre-trained Linear Deep Convolutional Network Class
    """
    def __init__(self, architecture=3, normalize_filters=None,
                 patch_shape=(7, 7)):
        super(PreTrainedLDCN, self).__init__(architecture=architecture)
        self.normalize_filters = normalize_filters
        self.patch_shape = patch_shape

    def build_network(self, filters):
        self._filters = _normalize_filters(filters, self.normalize_filters)
        self._n_filters = _parse_filters(self._filters)


