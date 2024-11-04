from setuptools import setup, find_packages

setup(
    name='agree-before-acting',  # Replace with your package's name
    version='0.1',
    packages=find_packages(),
    description="Code for the 'Agree Before Acting' paper",
    author="coldfrenzy",
    install_requires=[
        "numpy~=1.18.5",
        "torch~=1.7.0",
        "ray~=1.13.0",
        "wandb~=0.13.11",
        "argparse~=1.4.0",
        "Click~=7.0",
        "recordtype~=1.3",
        "matplotlib>=3.0.2",
        "msgpack==1.0.5",
        "msgpack-numpy~=0.4.4.0",
        "svgutils~=0.3.1",
        "pandas~=0.25.1",
        "importlib-resources>=1.0.1,<2",
        "timeout-decorator~=0.4.1",
        "attrs~=22.2.0",
        "gym==0.14.0",
        "networkx~=2.6.3",
        "graphviz~=0.20.1",
        "imageio~=2.26.0",
    ],  # Leave empty or add only internal requirements here
    include_package_data=True,
)