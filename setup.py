from setuptools import setup

setup(
  name='euclideanizing_flows',
  version='0.1.0',
  packages=['euclideanizing_flows'],
  url='',
  license='MIT',
  author='asif.rana',
  author_email='asif1253@gmail.com',
  description='stable dynamical systems using euclideanizing flows',
  install_requires = [
    'torch',
    'numpy',
    'matplotlib',
    'scipy'
  ]
)
