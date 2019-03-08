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
