from setuptools import setup, find_packages

setup(
    name="saferl-lite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    author="Your Name",
    description="A lightweight, explainable, and constrained reinforcement learning library",
    license="MIT",
)
