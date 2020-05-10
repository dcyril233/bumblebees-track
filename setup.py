# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bumblebees_track',
    version='0.1.0',
    description='tracking bumblebees through detecting the light spot on images',
    long_description=readme,
    author='Chunyu Deng',
    author_email='cdeng5@sheffield.ac.uk',
    url='https://github.com/dcyril233/bumblebees-track',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)