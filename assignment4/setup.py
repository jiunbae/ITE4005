from setuptools import setup, find_packages, Extension
from codecs import open
from os import path


import numpy as np

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '0.0.6'

# Get the long description from README.md
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    reqs = f.read().split('\n')

install_requires = [x.strip() for x in reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '')
                    for x in reqs if x.startswith('git+')]

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        'lib.algorithms.factorization',
        ['lib/algorithms/factorization' + ext],
        include_dirs=[np.get_include()]),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='recommender',
    author='maydev',
    author_email='alice.maydev@gmail.com',

    description=('ITE4005 assignment#4'),
    long_description=long_description,
    long_description_content_type='text/markdown',

    version=__version__,
    url='https://blog.maydev.org',

    license='GPLv3+',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
    keywords='recommender recommendation system',

    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    dependency_links=dependency_links,

    entry_points={'console_scripts':
                  ['surprise = surprise.__main__:main']},
)
