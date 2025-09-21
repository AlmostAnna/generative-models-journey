# setup.py
from setuptools import setup, find_packages

setup(
    name="generative-journey",
    version="0.1.0",
    description="Generative models exploration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scikit-learn",
        "matplotlib",
#        "imageio", # for GIF creation
    ],
    python_requires=">=3.8",
)

