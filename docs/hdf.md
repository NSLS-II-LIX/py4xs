#

## h5xs

  This is the base class that enables interaction with the h5 file that stores the experimental data, by
  associating the raw information in the h5 file into meanful py4xs objects (e.g. Data1d instances),
  on which data analysis can be carried out. It is assumed that the file structure follow 
  the convention: 
  
  * the data are organized by samples
  * under each sample, the raw data are stored under the `primary` group. Two datasets are expected: `data` and `timestamps`
  * the processed data are saved under the `processed` group. In this base class, processing is limited to
  azimuthal average. The datasets include those from the individual detectors, `merged` (from all detectors), `averaged` 
  (from multiple frames/exposures), and possibly `subtracted` (for background). 
  
  The file structure assciated with specific
  types of experiments supported at LiX are defined under `lixtools`. 
  
  Each `h5xs` object must be initilized with detector configurations and a $q$-grid.
  
  `fh5`: the file handle to the associated h5 file. `py4xs` now closes the file any time it is 
    not actively reading or writing data. Call `explicit_open()` to get the file handle. And close
    it afterward using `explicit_close()`.
  
  `d1s` : dictionaries of `Data1d` objects derived from the scattering patterns.
  
  `load_data()`: generate the initial `d1s` objects for each detector
  
  `merge_d1s()`: merge the processed data from individual detectors 
    
  `average_d1s()`: average together multiple scattering patterns for the same sample, based on a similarity
  analysis. Selection of the frames to be averaged together can be specified.
  
  `save_d1s()` and `load_d1s()`: save the `d1s` dictionaries to the h5 file from load them
  back from the h5 file.

  `plot_d1s()`: plot the 1D data from single sample, useful for showing how the frames are selected
  
  `compare_d1s()`: plot the 1D data from multiple samples, useful for visualizing the difference between samples

  `get_d2()`: return the `Data2d` object(s) corresponding to a specified sample name and frame number
  
  `get_d1()`: return the `Data1d` object corresponding to a specified sample name and frame number. By default
  returns the `merged` data.

  `get_mon()`: get the beam intensity monitor data corresponding to the individual scattering patterns. The 
  source of the intensity data is defined in `py4xs.local`.
  
  `show_data()`, `show_data_qphi()`, and `show_data_qxy()`: display the scattering pattern
  in the h5 file in various coordinates (see examples).

  `check_bm_center()`: take horizontal and vertical line cuts near the beam center and compare the intensity 
  on either side of the beam center, as a simplied means to verify whether the beam center specified in 
  the detector configuration is correct.

  `md_dict()`: return the meta data to be recorded in the exported ascii file, derived from the info 
  defined in `py4xs.local`. 
  
  `header()`: return the header of the Bluesky scan that produced the data  
  
---------------

## h5exp

`h5exp` is derived from `h5xs` and is used to store the detector configuration and $q$-grid 
  so that this information can be propagated easily to other `h5xs` objects.
  
  `save_detectors()` and `read_detectors()` are provided to exchange information between 
  a h5 file and the corresponding `h5exp` object. 
  
  `recalibrate()` : recalibrate the detector `exp_para` based on the scattering data collected
  from a standard sample (sivler behenate).
  
  
