from setuptools import setup
import py4xs

setup(
    name='py4xs',
    description="""python package for processing x-ray scattering data""",
    version=py4xs.__version__,
    author='Lin Yang',
    author_email='lyang@bnl.gov',
    license="MIT",
    url="",
    packages=['py4xs'],
    install_requires=['fabio', 'h5py', 'pillow', 
                      'matplotlib', 'numpy', 'scipy'],
    python_requires='>=3',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='x-ray scattering',
)
