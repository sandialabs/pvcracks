# setup.py
from setuptools import setup, find_packages

setup(
    name="pvcracks",
    version="0.1.0",
    description="PVCracks: crack‐induced power loss & crack progression in PV cells",
    author="Norman Jost",
    packages=find_packages(exclude=["docs", "*.ipynb_checkpoints"]),
    install_requires=[
        "numpy>=2.2.2",
        "pandas>=2.2.3",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "termcolor>=2.1.0",
        "matplotlib>=3.10.0",
        "scikit-image>=0.19.3",
        "imutils>=0.5.4",
        "opencv-python>=4.8.0",
        "xgboost>=3.0.2",
        "optuna>=4.2.1",
        "supertree>=0.5.5",
        "scikit-learn>=1.5.2",
        "requests>=2.28.0",
    ],
    python_requires=">=3.9",
)
