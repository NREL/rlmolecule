from typing import Any, List

import ray


@ray.remote
class RayDictCache:
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
class RaySetCache:
    def __init__(self):
        self._set = set()

    def add(self, key: Any):
        self._set.add(key)

    def contains(self, keys: List[Any]):
        return [key in self._set for key in keys]
