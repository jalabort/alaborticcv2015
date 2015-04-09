from .base import LinDeepConvNet


def parse_filters(filters):
    n_filters = []
    for fs in filters:
        n_filters.append(fs.shape[0])
    return n_filters


class PreTrainedLDCN(LinDeepConvNet):

    def __init__(self, filters):
        self._filters = filters
        self._n_filters = parse_filters(filters)
