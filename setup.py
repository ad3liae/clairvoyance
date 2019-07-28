# -*- coding: utf-8 -*-
import os
import sys
import shutil

from setuptools import setup, find_packages

metadata = dict(
  name='clairvoyance',
  version='0.1.0',
  description='Clairvoyance is a concurrent lip reader.',
  classifiers=[
    "Topic :: Security",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
  ],
  author='Ken-ya and Takahiro Yoshimura',
  author_email='takahiro_y@monolithworks.co.jp',
  url='https://github.com/monolithworks/clairvoyance',
  keywords='security',
)

README = open('README.rst').read()

setup(
  long_description=README,
  packages=find_packages(),
  package_data={'clairvoyance':[]},
  include_package_data=True,
  zip_safe=False,
  install_requires=[l for l in open('requirements.txt')],
  setup_requires=[
    "wheel",
  ],
  entry_points = {'console_scripts':['clairvoyance = clairvoyance.ui:shell']},
  **metadata
)
