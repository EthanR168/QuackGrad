from setuptools import setup, find_packages

setup(
    name="QuackGrad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy", 
    ],
    author="SirQuackPng",
    description="A simple auto-grad engine.",
    python_requires='>=3.7',
)