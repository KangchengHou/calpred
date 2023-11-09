from distutils.version import LooseVersion
from io import open

import setuptools
from setuptools import setup

setup(
    name="calpred",
    version="0.1",
    description="Calibrated prediction intervals",
    packages=["calpred"],
    setup_requires=["numpy>=1.10"],
    entry_points={"console_scripts": ["calpred=calpred.cli:cli"]},
    package_data={
        "calpred": ["calpred.cli.R"],
    },
    zip_safe=False,
)
