from setuptools import setup, find_packages

setup(
    name='umst-substrate-zoomer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy','pyyaml','matplotlib'],
    description='UMST Substrate Zoomer - multiscale resonance & binding visualization harness',
    author='Aerware AI',
)
