# py4xs

This is a python package for processing x-ray scattering data. It 
was first developed at the X9 beamline at NSLS. Since then, it has been 
revised multiple times and is now being used at the LiX beamline at NSLS-II. 
Compared to the previous versions, the major changes in this version are:

1. Revision of the conversion between detector images and the corresponding
intensity maps, with coordinates of qr-qz, q-phi, or just q. The conversions
are now realized using histograms. 

2. 2D plot of the data has been improved, with enhanced features to annotate 
the scattering intensity.

3. Added examples to illustrate the use of this module in various x-ray
scattering measurements. 



Changes since initial release on pyPI

2016-10-15:
minor bug fix in MatrixWithCoords.roi().

2016-10-20:
include geometric corrections in solution scattering 

2017:
bug fixes related to corrections applied to solution scattering data

2018-01: 
minor changes in Data2d.conv_Iqphi()

2018-02:
changes in Data2d and Data1d to prepare for hdf5 support

2018-07:
fixed a bug that caused uneven bins in Data2d.conv_Iq(); 
first version of hdf5 support

2018-09:
improved functionality and bug fixes for hdf support; 
bug fix for slnxs; 
fix azimuthal angular range to avoid large gap between values near Pi/-Pi;  
attach exp_para to hdf; 
removing qgrid from DetectorConfig (this affects the syntax of process/average/merge in slnxs); 

2018-10:
bugfixes, slnxs and hdf; 
handling of transmitted beam intensity;
first attempt to support scanning data 

2018-11:
slnxs bug fix; moved some functions from h5sol_HT to h5xs

2019-02:
uniform creation of data2d from either a filename or a numpy array;
h5exp for storing experimental configuration (no data);
export x-ray scattering chromatogram

2019-03:
add SVD background subtraction to h5sol_HPLC
return calculated chromtogram as data

2019-04:
delete existing "processed" data group when the length of qgrid changes
add mask items from code rather than from a file
error bar calculation in Data1d.avg()

2019-05:
added notebooks.py to define data processing notebook GUIs

2019-09:
meta data footer in exported 1D scattering profile

2019-10:
functions in h5sol_HT to compare samples and change buffer

2019-11:
fixed bug in bin_subtracted_frames(); added ATSAS report to HPLC GUI
handles em as monitor for transmission intensity;
create new h5 file with links to multiple files; update display_data_h5xs()

2020-01:
fixed tick lables in HPLC 2D plot

2020-02:
refined conv_Iq() to improve accuracy
functions to help generate mask from the data
h5xs.load_data() now work for scattering data store in 2D arrays

2020-03:
implemented cormap-equivalent pair-wise comparison
added check_bm_center() function in hdf

2020-04:
added estimate_scaling_factor() for buffer subtraction
assume default flow rate with older HPLC data
revised ATSAS support functions; paralellized modelling using Dask

2020-05:
revised h5xs.check_bm_center()
added h5exp.recalibrate()
incorporate estimate_scaling_dactor() into h5solHT.process()

2020-12:
revised handling of transField in h5xs
revised h5xs.load_data()
added h5xs.show_data_qxy() and show_data_qphi()


