from setuptools import setup

setup(
    name='py4xs',
    description="""python package for processing x-ray scattering data""",
    version="2018.07.24",
    author='Lin Yang',
    author_email='lyang@bnl.gov',
    license="MIT",
    url="",
    packages=['py4xs'],
    install_requires=['fabio', 'json', 'h5py', 'pillow',
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
