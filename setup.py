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
  install_requires=[
      'attrs',

      # LipNet
      'Keras==2.0.2',
      'editdistance==0.3.1',
      'h5py==2.6.0',
      'matplotlib==2.0.0',
      'numpy==1.12.1',
      'python-dateutil==2.6.0',
      'scipy==0.19.0',
      'Pillow==4.1.0',
      'tensorflow==1.0.0',
      'Theano==0.9.0',
      'nltk==3.2.2',
      'sk-video==1.1.10',
      'dlib==19.17.0'
  ],
  setup_requires=[
    "wheel",
  ],
  entry_points = {'console_scripts':['clairvoyance = clairvoyance.ui:shell']},
  **metadata
)
