from io import open
from os import path

from setuptools import find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='rlmolecule',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A package for deep reinforcement learning over tree-based search spaces',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NREL/rlmolecule',  # Optional
    author='Peter St. John',
    author_email='peter.stjohn@nrel.gov',  # Optional
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD 3-Clause',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),  # Required
    install_requires=[],  # TODO

    project_urls={
        'Source': 'https://github.com/NREL/rlmolecule',
    },
)
