#!/usr/bin/env python3

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]


setup(
    name='slurmhelper',
    version='1.1',
    description='Small helper tool to create and submit SLURM jobs.',
    author='David Dormagen',
    author_email='czapper@gmx.de',
    url='https://github.com/walachey/slurmhelper/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['slurmhelper'],
    package_dir={'slurmhelper': 'slurmhelper/'}
)
