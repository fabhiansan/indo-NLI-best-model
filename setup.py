from setuptools import setup, find_packages

setup(
    name="indo-nli-model",
    version="0.1.0",
    packages=find_packages(),
    description="A project for building and evaluating NLI models on the IndoNLI dataset",
    author="Cascade User",
    author_email="",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
    ],
    python_requires=">=3.7",
)
