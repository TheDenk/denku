#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the denku package.
"""

import os
import re
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
        print(f"Warning: Could not read file {filename}")
        return ""


def get_version():
    """Extract the version from the package's __init__.py file.
    
    Returns:
        str: The package version.
    """
    init = read_file('denku/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])
    return "0.1.0"


def get_requirements():
    """Get the list of requirements from requirements.txt.
    
    Returns:
        list: List of requirements.
    """
    try:
        requirements_file = read_file('requirements.txt')
        if requirements_file:
            return [line.strip() for line in requirements_file.splitlines() 
                    if line.strip() and not line.startswith('#')]
    except Exception as e:
        print(f"Warning: Error reading requirements.txt: {e}")
    
    # Fallback to default requirements
    return [
        'numpy>=1.11.1',
        'Pillow>=9.5.0',
        'matplotlib>=3.6.0',
        'opencv-python>=4.7.0.72',
        'opencv-python-headless>=4.6.0.66',
        'opencv-contrib-python>=4.6.0.66',
        'torch>=1.7.0',
    ]


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
    install_requires=get_requirements(),
    python_requires='>=3.7',
    project_urls={
        'Bug Reports': f'{URL}/issues',
        'Source': URL,
    },
)