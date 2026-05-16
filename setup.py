# setup.py
from setuptools import setup, find_packages

setup(
    name="attractiveness_api",
    version="1.0.0",
    description="Facial Attractiveness Prediction API using Deep Learning (MobileNetV2, EfficientNet, ResNet)",
    author="Nisar Ahmad",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # keep minimal; refer requirements.txt
    ],
)
