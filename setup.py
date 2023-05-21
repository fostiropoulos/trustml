#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="trustml",
    version="1.0",
    description="Supplementary for Trustworthy model evaluation on a budget",
    author="Iordanis Fostiropoulos",
    author_email="dev@iordanis.xyz",
    url="https://iordanis.xyz/",
    python_requires=">3.10",
    long_description=open("README.md").read(),
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "ablator @ git+https://github.com/fostiropoulos/ablator.git",
        "emukit==0.4.10",
        "gdown==4.7.1",
        "matplotlib==3.7.1",
        "nats_bench==1.8",
        "numpy==1.24.1",
        "omegaconf==2.2.3",
        "optuna==3.1.1",
        "pandas==2.0.0",
        "ray>2.1.0",
        "scikit_learn==1.2.2",
        "scipy==1.10.1",
        "seaborn==0.12.2",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "pyDOE==0.3.8",
        "Jinja2==3.1.2"
    ],
    extras_require={
        "dev": [
            "mypy==1.2.0",
            "pytest==7.3.0",
            "pylint==2.17.2",
            "flake8==6.0.0",
            "black==23.3.0",
            "types-requests==2.28.11.17",
        ],
    },
)
