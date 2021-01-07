.. py4xs documentation master file, created by
   sphinx-quickstart on Tue Jun  4 15:12:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

py4xs documentation
=================================

py4xs is a collection of python modules that are developed to help process x-ray 
scattering data. It provides the following functionalities:

* **Recipracal coordinates translation** Once the scattering geometry is defined,
  the recipracal coordinates for each detector pixel is calculated. Functions are 
  provided to translate the data into maps of various coordinates.

* **Plotting** Scattering data are plotted using matplotlib with awareness of 
  scattering 

* **Solution scattering** Support for azimuthal average, merging of data from 
  multiple detectors, and buffer scattering subtraction.

* **Data processing GUIs** These are provided as functions that run within Jupyter
  notebooks.

* **HDF5 packaging** All information relevant to the scattering experiment, including
  scattering geometry, raw data, and processed data, can be packaged into a single
  hdf5 file.

* **Data processing pipeline (under development)** Custom data processing pipelines 
  can be defined and saved in the hdf5 file.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples
   py4xs-classes-overview
   jupyter-notebook-GUIs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
