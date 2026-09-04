from setuptools import setup, find_packages

setup(
    name="cryogmm",
    version="0.1.0",
    description="Cryo-EM Gaussian Mixture Model",
    author="Minhuan Li",
    author_email="minhuanli@flatironinstitute.org",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "torch",
        "mdtraj",
        "scikit-learn",
        "fpsample",
        "tqdm",
        "matplotlib",
    ],
    url="https://github.com/minhuanli/cryoGMM",
)
