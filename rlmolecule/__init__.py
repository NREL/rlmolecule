from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

import lazy_import
lazy_import.lazy_module("tensorflow")