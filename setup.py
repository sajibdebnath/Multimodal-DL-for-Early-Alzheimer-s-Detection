from setuptools import setup, find_packages

setup(
    name="mdlf-alzheimer",
    version="1.0.0",
    description="Multimodal Deep Learning Framework for Early Alzheimer's Disease Detection",
    author="Sajib Debnath et al.",
    packages=find_packages(exclude=["scripts", "configs", "data/raw"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "timm>=0.9.2",
        "scikit-learn>=1.2.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "nibabel>=5.1.0",
        "scipy>=1.11.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "shap": ["shap>=0.42.1"],
        "hpo":  ["optuna>=3.2.0"],
        "survival": ["lifelines>=0.27.0"],
    },
)
