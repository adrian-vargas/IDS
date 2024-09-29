from setuptools import setup, find_packages

setup(
    name='IDS',
    version='0.1.0',
    description='Implementación del algoritmo Interpretable Decision Sets (IDS)',
    author='Tu Nombre',
    author_email='tu.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'apyori',
        'scikit-learn',
        'imblearn',
        'pulp',
    ],
    python_requires='>=3.6',
)
