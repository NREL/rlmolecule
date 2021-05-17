from collections import defaultdict, Callable
from functools import wraps
from time import time


class CallMetrics(object):
    def __init__(self):
        self.execution_time = defaultdict(lambda: 0.)
        self.execution_count = defaultdict(lambda: 0)

    def reset(self):
        self.execution_time = defaultdict(lambda: 0.)
        self.execution_count = defaultdict(lambda: 0)

    @property
    def data(self):
        return {'execution_time': dict(self.execution_time), 'execution_count': dict(self.execution_count)}


call_metrics = CallMetrics()


def collect_metrics(func: Callable):
    """Wraps the function to store total call time and number of calls in call_metrics.

    :param func: Function to be wrapped
    :return: Wrapped function that stores execution information to call_metrics.
    """
    @wraps(func)
    def store_metrics(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        call_metrics.execution_count[func.__name__] += 1
        call_metrics.execution_time[func.__name__] += end_time - start_time
        return result

    return store_metrics
