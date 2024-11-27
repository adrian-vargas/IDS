#IDS/setup.py

from setuptools import setup, find_packages

setup(
    name='IDS',
    version='0.1.0',
    description='Librería en Python para la implementación de modelos Interpretable Decision Sets (IDS)',
    author='Adrian Vargas',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'apyori',
        'scikit-learn',
        'imbalanced-learn',
        'pulp',
        'matplotlib',
        'graphviz',
        'pillow',
    ],
    python_requires='>=3.10.12',
)
