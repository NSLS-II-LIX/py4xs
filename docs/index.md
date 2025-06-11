#
py4xs is a collection of python modules for processing x-ray 
scattering data. It helps the experimenter to translate the experimental data from raw detector 
images to a form that can be further analyzed using generic numerical tools, by taking into
account the configuration of the x-ray scattering instrumentation. 

The development of py4xs started at the X9 beamline at NSLS and continued at 
the LiX beamline after the transition to NSLS-II, where it is being actively used for processing 
data in solution scattering and microbeam mapping experiments.

py4xs provides the following generic functionalities:

* **Reciprocal coordinates translation** Once the scattering geometry is defined,
  the reciprocal coordinates for each detector pixel is calculated. Functions are 
  provided to translate the data into maps of various coordinates.

* **Plotting** Scattering data are plotted using matplotlib with awareness of 
  the experimental configuration

* **Solution scattering** Support for azimuthal average, merging of data from 
  multiple detectors, and buffer scattering subtraction.

* **HDF5 packaging** All information relevant to the scattering experiment, including
  scattering geometry, raw data, and processed data, can be packaged into a single
  hdf5 file. For now it is assumed that the hdf layout is based on export from the 
  nsls2 databroker.

## py4xs project layout

    README.md        
    py4xs/
        exp_para.py  # experimental parameters
        detector_config.py
                     # detector configuration
        data2d.py    # functions related to 2d data
        slnxs.py     # functions related to 1d data
        mask.py      # mask used for azimuthal average
        plot.py      # functions for plotting 2D data
        hdf.py       # python interface to the hdf5 data file 
        utils.py     # utility functions
    doc/             
                     # documentation pages

In principle py4xs can be used to process scattering data collected on any instruments, as long
as the detector configurations are defined. Functionalities specific to data collected at the LiX 
beamline are implemented under lixtools, to provide the class definition and processing GUI for 
the supported types of experiments: 

## lixtools project layout

    README.md        
    lixtool/
        hdf/
            sol.py         # python classes for static solution scattering data
            hplc.py        # python class for in-line SEC data
            an.py          # base class for processed data stored separately from raw data 
            scan.py        # python class for scanning imaging data
        notebooks/
            generic.py     # GUI for generic solution scatterintg/powder diffraction data
            sol_static.py  # GUI for static solution scattering
            sol_hplc.py    # GUI for in-line SEC
            scanning.py    # GUI for scanning mapping and tomography
        mapping/
            common.py      # functions for structural mapping
            plants.py      # functions specific to scattering data from plant samples
        tomo/
            common.py      # functions for tomography
            FLcorrections.py
                           # functions for XRF absorption correction, from Mingyuan Ge
        inst/
            check_deck_config2.py
                           # script for sample prep using Opentrons OT2, runs on the RasPi 
            ot2_gen_prot.py
                           # script for reading the QR codes and generate transfer protocol
                           # run from data directory
            webcam.py      # read QR codes from a webcam




