#

## h5xs

  This is the base class that associates a h5 file with scattering data. `h5xs` translates 
  the raw information in the h5 file into meanful py4xs objects (e.g. Data1d instances),
  on which data analysis can be carried out.
  
  Each `h5xs` object must be initilized with detector configurations and a $q$-grid.
  
  ``fh5``: the file handle to the associated h5 file. 
  
  ``d1s`` : dictionaries of Data1d objects derived from the scattering patterns, organized
  by samples, then by data source.
  
  ``load_data()``: generate the initial `d1s` objects from the scattering patterns
  
  `save_d1s()` and `load_d1s()`: save the `d1s` dictionaries to the h5 file from load them
  back from the h5 file.
  
  `show_data()`, `show_data_qphi()`, and `show_data_qxy()`: display the scattering pattern
  in the h5 file in various coordinates (see examples).
  
---------------

## h5exp

`h5exp` is derived from `h5xs` and is used to store the detector configuration and $q$-grid 
  so that this information can be propagated easily to other `h5xs` objects.
  
  ``save_detectors()`` and ``read_detectors()`` are provided to exchange information between 
  a h5 file and the corresponding `h5exp` object. 
  
  ``recalibrate()`` : recalibrate the detector `exp_para` based on the scattering data collected
  from a standard sample (sivler behenate).
  
  
