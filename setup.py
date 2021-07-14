from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gvp',
    packages=find_packages(include=[
        'gvp',
        'gvp.data',
        'gvp.models'
    ]),
    version='0.1.1',
    description='Geometric Vector Perceptron',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'torch',
        'torch_geometric',
        'torch_scatter',
        'torch_cluster',
        'tqdm',
        'numpy',
        'sklearn',
        'atom3d'
    ]
)