from typing import Any, List

import ray
from lru import LRU


class DictCache:
    def __init__(self):
        self._dict = {}

    def put(self, key, value):
        self._dict[key] = ray.put(value)

    def get(self, key):
        ref = self._dict.get(key, None)
        if ref is not None:
            return ray.get(ref)
        else:
            return None


@ray.remote
class RayDictCache(DictCache):
    pass


@ray.remote
class RayLRUCache(DictCache):
    def __init__(self, max_size: int = int(1e5)):
        self._dict = LRU(max_size)


@ray.remote
class RaySetCache:
    def __init__(self):
        self._set = set()

    def add(self, key: Any):
        self._set.add(key)

    def contains(self, keys: List[Any]):
        return [key in self._set for key in keys]


def get_builder_cache(max_size: int = int(1e5)):
    return RayLRUCache.options(
        name="builder_cache", lifetime="detached", get_if_exists=True
    ).remote(max_size)


def get_terminal_cache():
    return RaySetCache.options(
        name="terminal_cache", lifetime="detached", get_if_exists=True
    ).remote()
