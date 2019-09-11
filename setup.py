#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import io
import re
from setuptools import setup, find_packages

with open('README_oneshot.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY_oneshot.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Amit Aides",
    author_email='amitaid@il.ibm.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Experiments in One-Shot deeplearning",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='oneshot',
    name='oneshot',
    packages=find_packages(include=['oneshot']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/amitaid/oneshot',
    version='0.1.0',
    zip_safe=False,
)


with open('README_experiment.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY_experiment.rst') as history_file:
    history = history_file.read()

requirements = ['py3nvml', 'traitlets']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Amit Aides",
    author_email='amitaid@il.ibm.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Framework for running experiments.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='experiment',
    name='experiment',
    packages=find_packages(include=['experiment']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.ibm.com/AMITAID/experiment',
    version='0.2.0',
    zip_safe=False,
)

def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read('README_ignite.rst')

requirements = ['enum34;python_version<"3.4"', 'futures; python_version == "2.7"', 'torch']

setup(
    # Metadata
    name='ignite',
    version='0.1.2',
    author='PyTorch Core Team',
    author_email='soumith@pytorch.org',
    url='https://github.com/pytorch/ignite',
    description='A lightweight library to help with training neural networks in PyTorch.',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('tests', 'tests.*',)),

    zip_safe=True,
    install_requires=requirements,
)

