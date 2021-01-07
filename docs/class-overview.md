py4xs classes overview
*****************************

This section describes the classes defined in py4xs, and their attributes and methods 
that are important for data processing. More detailed information can be found in the 
docstrings.

ExpPara
---------------
  Experimental parameters necessary to define the scattering geometry for 
  a 2D detector. 

  ``bm_ctr_x, bm_ctr_y``  specify the beam center position, defined as the pixel coordinates 
  of the intersection between the beam and the detector.

  ``ratioDw`` is the sample-to-detector distance, defined as a dimensionless ratio between 
  this distance and the horizontal dimenson of the detector sensing area.

  ``wavelength`` is the x-ray wavelength, in angstrom ( :math:`{\unicode{x212B}}` ) .

  ``flip`` defines how the detector image should be reoriented when read from the data file.
  The numberical value is the number of 90-deg rotations. The image is first mirrored 
  (lef and right) if the value is negative. 

  ``rot_matrix`` is the rotation matrix that translate the detector pixel position into 
  coordinates in the lab reference frame. This matrix should be calculated in classes 
  inhereted from ExpPara, where the method of calculatiing this rotation matrix is defined.

  ``sample_normal, incident_angle`` are parameters that define the sample orientation in grazing incident geometry ( only effective when ``grazing_incidence`` is `True` ). 

  From these parameters, these reciprocal coordinates for each pixel are then calculated:

  ``Q`` : the amplitude of the scattering vector, in :math:`{\unicode{x212B}}^{-1}`

  ``Phi`` : the azimuthal angle, :math:`\phi`, in degrees.

  ``Qr, Qn`` : projection of the scattering vector perpendicular/paralell to the sample
  normal (grazing incidence)

  ``xQ, yQ``: projection of the scattering vector in the horizontal/vertical directions.

  These common correction factors are also calculated:

  ``FPol`` is the polarization vector, assuming that the x-ray is linearly polarized along
  the horizontal direction (synchrotron radiation).

  ``FSA``  accounts for the difference in the solid angle each pixel opens to the sample.

Mask
---------------
  Mask for the scattering data to eliminate pixels to be excluded from data 
  processing. A mask in py4xs is typically defined as a collection of geometric
  shapes. The mask should have the same dimension as the detector image.

  ``map`` : the actual bit map that indicate excluded pixels

  ``read_file()`` : read the mask from a text file that defines the collection
  of shapes

  ``read_from_8bit_tif()`` : create the mask map from a 8-bit tif file. All the pixels 
  with the numerical value of 255 will be considered as part of the mask. This is useful 
  to be used together with ``Data2d.save_as_8bit_tif()``, which creates a tif file from 
  the 2D data. The tif file can be then modified in an image editor like gimp, where 
  the mask is painted white.

  ``reload()`` : re-read the text mask that defines the mask. Any addition to the 
  mask after the last read will be lost.

  ``add_item()`` : add new shapes into the existing mask


DetectorConfig
---------------
  All detector-related information

  ``extension`` : file extension for the data file

  ``exp_para`` : experimental parameters for the detector, as defined above

  ``mask`` : as defined above

  ``dark, flat`` : dark field and flat field data needed to apply corrections.

  ``pre_process()`` : method for performing drak field and flat feld corrections.  


MatrixWithCoords
----------------
  This class is created to define intensity maps, for which the coordinates for
  the pixels in the map need to be defined. 

  ``d`` : the actual intensity map

  ``xc, yc`` : 1d arrays that define the x and y coordinates of the intensity map

  ``datatype`` : type of the data (e.g. detector raw image, :math:`I(q, \phi)` map, ...), which 
  must be of type Enum `DataType`. This is useful when plotting the intensity map,
  so that the correct coordinates can be displayed. 

  ``conv(self, Nx1, Ny1, xc1, yc1, mask=None, cor_factor=1, datatype=DataType.det)`` 
  converts the intensity map to new coordinates xc1, yc1. Masked pixels are omitted.

  ``line_cut(self, x0, y0, ang, lT, lN, nT, nN, mask=None)`` 
  returns a line cut from the intensity map. Masked pixels are omitted.

Data2d
----------------
  Generic scattering data. Multiple representations can exist: ``data`` (raw detector
  image), ``qrqz_data`` ( :math:`q_r - q_z` map, grazing incidence), and ``qphi_data`` 
  ( :math:`q - \phi` map).

  ``exp`` : the ExpPara for the scattering data.

  ``label`` : by default this is set to the name of the data file. But it could be set 
  when the instance is initilized.  

  ``uid, timestamp`` : optional information. They are extracted from the data file
  for Pilatus data collected at NSLS-II.  

  ``conv_Iq(self, qgrid, mask=None, cor_factor=1)`` converts the 2D scattering data
  into a 1D scattering intensity profile.

  ``conv_Iqphi()`` and ``conv_Iqrqz()`` generate the :math:`I(q, \phi)` and :math:`I(q_r, q_z)` 
  map of the scattering data.


Axes2dPlot
------------

  This class displays 2D scattering data in a matplotlib ``Axes``. It captures mouse
  clicks to display the reciprocal coordinates at the clicked pixel, and can overlay
  `decorations` (lines and symbols at specified reciprocal coordinates) onto the scattering data.

  ``ax, cmap`` : parameters used by matplotlib to plot the sample

  ``d2, exp`` : the scattering data and its ExpPara.

  ``mark_points()`` , ``mark_coords()`` and ``mark_standard()`` generate the decorations 
  (points, lines/grids with the given coordinates, and powder rings expected from standard
  samples) to be overlaid onto the scattering pattern. Refer to the Examples section.

  ``plot()`` plots the data. 


Data1d
-----------

  ``trans`` is the (relative) value of the transmitted intensity, which is used as a reference
  for normlization during background subtraction. Two modes are allowed (defined as members of the 
  `Enum` `transMode` ) : `external` or `from_waxs`. For `transMode.external` , the `trans`
  value must be specified explicitly. For `transMode.from_waxs`, the `trans` value is calculated
  from the water scattering peak intensty near :math:`2.0 \unicode{x212B}^{-1}` .

  ``load_from_2D()`` populates the atrtibutes of the instance based on the input data, which can be
  a Data2d object, a data file, or a numpy array, and the specified ExpPara. The azimuthally averaged
  1D data is generated using Data2d.conv_Iq(), after applying the polorization and solid angle 
  corrections. The error bar is the standard deviation of intensity in all pixels that belong to the
  same *q* value.

  ``merge()`` merges data with another Data1d object. Within the overlapping *q*-range, the 
  scattering intensity is averaged and the original data are saved can could be displayed later.

  ``avg()`` performs averaging with the given set of Data1d objects. 

  ``bkg_cor()`` performs background subtraction, based on the *trans* value.

  ``plot()`` plots the data, in a given matplotlib Axes if specified.

  ``save()`` exports the data into a text file in a 3-column format (*q*, intensity, error bar).

  ``plot_Guinier()`` plots :math:`ln[I(q)]` vs :math:`q^2` (Guinier plot) and report :math:`I_0` and 
  :math:`R_g` .


