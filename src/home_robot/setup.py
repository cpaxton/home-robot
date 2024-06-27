# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy==1.26.4",
    "scipy==1.14.0",
    "hydra-core",
    "yacs",
    "pybullet",
    "pygifsicle",
    "open3d==0.18.0",
    "scikit-image",
    "pybind11-global",
    "sophuspy==1.2.0",
    "trimesh==4.4.0",
    "pin>=2.6.17",
    "pillow==9.5.0",  # For Detic compatibility
]

setup(
    name="home-robot",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)