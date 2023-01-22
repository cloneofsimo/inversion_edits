import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="inversion_edits",
    py_modules=["inversion_edits"],
    version="0.1.0",
    description="Inversion Edits",
    author="Simo Ryu",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
