from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="Lin2025",
    version="0.0.0",
    description='Code base for "Self-Attention-Based Contextual Modulation Improves Neural System Identification", Lin et al. 2020',
    author="Isaac Lin",
    author_email="isaacl@cs.cmu.edu",
    packages=find_packages(exclude=[]),
    install_requires=["torch>=2.0", "pandas", "tqdm", "requests"],,
)
