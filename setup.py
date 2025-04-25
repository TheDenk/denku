#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the denku package.
"""

import os
from setuptools import setup, find_packages


def read_file(filename):
    """Read the contents of a file.

    Args:
        filename (str): Path to the file relative to the setup.py directory.

    Returns:
        str: Contents of the file.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
            return f.read()
    except (IOError, FileNotFoundError):
        print(f'Warning: Could not read file {filename}')
        return ''


def get_version():
    """Extract the version from the package's __init__.py file.

    Returns:
        str: The package version.
    """
    init = read_file('denku/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])
    return '0.1.0'


def get_requirements(dev=False):
    """Get the list of requirements from requirements.txt or requirements-dev.txt.

    Args:
        dev (bool): If True, include development requirements.

    Returns:
        list: List of requirements.

    Raises:
        FileNotFoundError: If requirements file is not found.
        ValueError: If requirements file is empty or invalid.
    """
    requirements_file = 'requirements-dev.txt' if dev else 'requirements.txt'
    requirements = read_file(requirements_file)

    if not requirements:
        raise FileNotFoundError(f'{requirements_file} not found or is empty')

    # Parse requirements, handling -r references
    parsed_requirements = []
    for line in requirements.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r'):
            # Recursively get requirements from referenced file
            ref_file = line[2:].strip()
            parsed_requirements.extend(get_requirements(
                ref_file == 'requirements-dev.txt'))
        else:
            parsed_requirements.append(line)

    if not parsed_requirements:
        raise ValueError(f'No valid requirements found in {requirements_file}')

    return parsed_requirements


# Package metadata
NAME = 'denku'
DESCRIPTION = 'Custom computer vision utilities for image and video processing, visualization, and memory management.'
URL = 'https://github.com/TheDenk/denku'
AUTHOR = 'Karachev Denis'
AUTHOR_EMAIL = 'welcomedenk@gmail.com'
LICENSE = 'Apache'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Image Processing'
]

# Setup configuration
setup(
    name=NAME,
    version=get_version(),
    description=DESCRIPTION,
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords='computer-vision, image-processing, video-processing, visualization, opencv, numpy, pytorch',
    packages=find_packages(),
    install_requires=get_requirements(dev=False),
    extras_require={
        'dev': get_requirements(dev=True),
    },
    python_requires='>=3.8',
    project_urls={
        'Bug Reports': f'{URL}/issues',
        'Source': URL,
    },
)
