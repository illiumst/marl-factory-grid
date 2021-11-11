from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = [line.strip() for line in f]

setup(
    name="EDYs",
    version="0.1",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=reqs,
)