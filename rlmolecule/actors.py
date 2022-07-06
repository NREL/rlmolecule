import ray


@ray.remote
class RayCache:
    def __init__(self):
        self._dict = {}

    def put(self, key, value):
        self._dict[key] = ray.put(value)

    def get(self, key):
        return self._dict.get(key, None)
