from .base import LinDeepConvNet, _parse_filters, _normalize_filters


class PreTrainedLDCN(LinDeepConvNet):
    r"""
    Pre-trained Linear Deep Convolutional Network Class
    """
    def build_network(self, filters):
        self._filters = _normalize_filters(filters, self.norm_func)
        self._n_filters = _parse_filters(self._filters)


