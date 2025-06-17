from setuptools import setup, find_packages

setup(
    name="saferl-lite",
    version="0.1.2",
    author="Satyam Mishra",
    author_email="satyam@example.com",
    description="A lightweight, explainable, and constrained reinforcement learning toolkit.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/satyamcser/saferl-lite",
    project_urls={
        "Documentation": "https://github.com/satyamcser/saferl-lite/tree/main/docs",
        "Source": "https://github.com/satyamcser/saferl-lite",
        "Bug Tracker": "https://github.com/satyamcser/saferl-lite/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True,
)
