"""
Setup script for Synthetic Payment Data Generator
Script de instalación para Generador de Datos de Pago Sintéticos
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="synthetic-payments-generator",
    version="1.0.0",
    author="MaxPelu",
    description="AI-powered synthetic B2B payment data generator using GANs and VAEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaxPelu/Synthetic-Payments-Data-Generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "synpay-generate=src.data_generator:main",
            "synpay-eda=src.eda_analysis:main",
            "synpay-train=src.train_pipeline:main",
            "synpay-example=src.example_usage:main",
        ],
    },
)
