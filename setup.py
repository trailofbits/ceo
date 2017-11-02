#!/usr/bin/python2
from setuptools import setup

setup(
    name='CEO',
    version='0.1',
    packages=['ceo'],
    include_package_data=True,
    description='',
    long_description="",
    url='https://github.com/ggrieco-tob/ceo',
    author='Trail of Bits',
    scripts=[
        'ceo-bin'
        ],
    install_requires=[
        "scipy",
        "scikit-learn",
        "imbalanced-learn"
       ],
)
