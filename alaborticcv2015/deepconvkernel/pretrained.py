from .base import LinDeepConvNet, _parse_filters


class PreTrainedLDCN(LinDeepConvNet):

    def __init__(self, filters):
        self._filters = filters
        self._n_filters = _parse_filters(self._filters)
