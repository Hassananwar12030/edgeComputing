"""
Setup script for the Federated Learning Sensor Fusion project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thesis-federated-sensor-fusion",
    version="0.1.0",
    author="Adnan Anwar Rajput",
    author_email="",
    description="Federated Learning for Audio-Visual Sensor Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "paho-mqtt>=1.6.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "edge": [
            "librosa>=0.10.0",
            "sounddevice>=0.4.6",
            "ultralytics>=8.0.0",
            "flwr>=1.0.0",
        ],
        "server": [
            "tensorflow>=2.10.0",
            "flwr>=1.0.0",
            "librosa>=0.10.0",
            "matplotlib>=3.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thesis-edge=src.edge.main:main",
            "thesis-server=src.server.main:main",
            "thesis-flower=src.server.flower_server:main",
        ],
    },
)
