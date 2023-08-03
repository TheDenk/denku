# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        file_content = f.read()
    return file_content


def get_version():
    init = read('denku/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])


setup(
    name='denku',
    version=get_version(),
    description='Custom CV functions.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/TheDenk/denku',
    author='Karachev Denis',
    author_email='welcomedenk@gmail.com',
    license='Apache',
    install_requires=['numpy>=1.11.1', 'Pillow>=9.5.0', 'matplotlib>=3.6.0', 'opencv-python>=4.7.0.72', 'opencv-python-headless>=4.6.0.66', 'opencv-contrib-python>=4.6.0.66'],
    packages=find_packages(),
)