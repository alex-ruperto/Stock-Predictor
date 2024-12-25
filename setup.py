from setuptools import setup, find_packages

setup(
    name="stock-predictor",
    version="1.0.0",
    description="A stock prediction tool that uses machine learning to predict stock prices.",
    author="Alex Ruperto",
    packages=find_packages(), # Automatically finds all of the submodules.
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pandas-ta",
        "backtrader",
        "alpaca-trade-api",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)