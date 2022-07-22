import atexit
import csv
import logging
from typing import Any, List

import ray
from lru import LRU

logger = logging.getLogger(__name__)


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
        if not str(key).endswith(" (t)"):
            logger.info(f"pruning non-terminal state {key}")
        self._set.add(key)

    def contains(self, keys: List[Any]):
        return [key in self._set for key in keys]


@ray.remote
class CSVActorWriter:
    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._filehandle = open(self._filename, "wt", newline="", buffering=1)
        self._writer = csv.writer(self._filehandle)
        atexit.register(lambda: self.close())

    def write(self, row):
        self._writer.writerow(row)

    def close(self):
        self._filehandle.close()


def get_builder_cache(max_size: int = int(1e5)):
    return RayLRUCache.options(
        name="builder_cache", lifetime="detached", get_if_exists=True
    ).remote(max_size)


def get_terminal_cache():
    return RaySetCache.options(
        name="terminal_cache", lifetime="detached", get_if_exists=True
    ).remote()


def get_csv_logger(filename: str):
    return CSVActorWriter.options(
        name="csv_logger", lifetime="detached", get_if_exists=True
    ).remote(filename)
