#!/usr/bin/env python3
"""
Setup script for media_manager package.
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='media_manager',
    version='0.1.0',
    description='Media manager - like git for your media files',
    author='Media Manager Authors',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'media=media_manager.media:main',
        ],
    },
    python_requires='>=3.6',
)