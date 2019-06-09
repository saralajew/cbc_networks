# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

long_description = '''
CBC networks in Keras.
'''

setup(name='keras_cbc',
      version='0.0.1',
      description='CBC networks in Keras.',
      long_description=long_description,
      author='CBC authors',
      author_email='cbc.authors@gmail.com',
      url='https://github.com/cbc-authors/cbc_networks',
      download_url='https://github.com/cbc-authors/cbc_networks.git',
      license='BSD 3-Clause License',
      install_requires=['keras==2.2.4',
                        'opencv-python==4.1.0.25',
                        'Pillow==6.0.0',
                        'tensorflow-gpu==1.12.0'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
