#
py4xs is a collection of python modules for processing x-ray 
scattering data. It helps the experimenter to translate the experiemntal data from raw detector 
images to a form that can be further analyzed using generic numerical tools, by taking into
account the configuration of the x-ray scattering instrumentation. 

The development of py4xs started at the X9 beamline at NSLS and continuted at 
the LiX beamline after the transition to NSLS-II, where it is being actively used for processing 
data in solution scattering and microbeam mapping experiments.

py4xs provides the following generic functionalities:

* **Recipracal coordinates translation** Once the scattering geometry is defined,
  the recipracal coordinates for each detector pixel is calculated. Functions are 
  provided to translate the data into maps of various coordinates.

* **Plotting** Scattering data are plotted using matplotlib with awareness of 
  the experimental configuration

* **Solution scattering** Support for azimuthal average, merging of data from 
  multiple detectors, and buffer scattering subtraction.

* **HDF5 packaging** All information relevant to the scattering experiment, including
  scattering geometry, raw data, and processed data, can be packaged into a single
  hdf5 file.

* **Data processing pipeline (under development)** Custom data processing pipelines 
  can be defined and saved in the hdf5 file.

Functionalities specific to the LiX beamline (e.g. data processing GUIs) can be found
under lixtools.

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

