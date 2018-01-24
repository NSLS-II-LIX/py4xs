from setuptools import setup

setup(
    name='py4xs',
    description="""python package for processing x-ray scattering data""",
    version="2017.12.20",
    author='Lin Yang, Hugo Slepicka',
    author_email='lyang@bnl.gov',
    license="MIT",
    url="",
    packages=['py4xs'],
    install_requires=['fabio', 
                      'matplotlib', 
                      'numpy', 
                      'scipy', 
                      'pillow'],
    python_requires='>=3',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
    ],
    keywords='x-ray scattering',
)
