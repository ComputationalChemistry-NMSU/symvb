from setuptools import setup
import os


def get_long_description():
    topdir = os.path.abspath(os.path.join(__file__, '..'))
    with open(os.path.join(topdir, 'README.md'), 'r') as f:
        return f.read()


def get_version():
    topdir = os.path.abspath(os.path.join(__file__, '..'))
    with open(os.path.join(topdir, 'symvb', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise ValueError("Version string not found")

VERSION = get_version()

setup(name='symvb',
      version=VERSION,
      description='Symbolic calculations for semi-quantitative Valence Bond Theory',
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      url='https://github.com/ComputationalChemistry-NMSU/symvb',
      author='Marat Talipov',
      author_email='talipovm@nmsu.edu',
      license='MIT',
      license_files=['LICENSE'],
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
      packages=['symvb'],
      zip_safe=True,
      python_requires='>=3.8',
      # numpy/scipy are imported at module level by molecule, functions,
      # numerical, symmetry, spin, operators, and mo_projection
      install_requires=['sympy>=1.13', 'numpy>=1.24', 'scipy>=1.10'],
      )