# py4xs overview

py4xs is a collection of python modules that are developed for processing x-ray 
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

## Project layout

    README.md        
    py4xs/
        exp_para.py  # experimental parameters
        detector_config.py
                     # detector configuration
        data2d.py    # functions related to 2d data
        slnxs.py     # functions related to 1d data
        hdf.py       #
    doc/             
                     # documentation pages

