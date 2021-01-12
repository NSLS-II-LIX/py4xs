#
## MatrixWithCoords
  This class is created to define intensity maps, for which the coordinates for
  the pixels in the map need to be defined. 

  ``d`` : the actual intensity map

  ``xc, yc`` : 1d arrays that define the x and y coordinates of the intensity map

  ``datatype`` : type of the data (e.g. detector raw image, $I(q, \phi)$ map, ...), which 
  must be of type Enum `DataType`. This is useful when plotting the intensity map,
  so that the correct coordinates can be displayed. 

  ``conv(self, Nx1, Ny1, xc1, yc1, mask=None, cor_factor=1, datatype=DataType.det)`` 
  converts the intensity map to new coordinates xc1, yc1. Masked pixels are omitted.

  ``line_cut(self, x0, y0, ang, lT, lN, nT, nN, mask=None)`` 
  returns a line cut from the intensity map. Masked pixels are omitted.

----------------

## Data2d
  Generic scattering data. Multiple representations can exist: ``data`` (raw detector
  image), ``qrqz_data`` ( $q_r - q_z$ map, grazing incidence), and ``qphi_data`` 
  ( $q - \phi$ map).

  ``exp`` : the ExpPara for the scattering data.

  ``label`` : by default this is set to the name of the data file. But it could be set 
  when the instance is initilized.  

  ``uid, timestamp`` : optional information. They are extracted from the data file
  for Pilatus data collected at NSLS-II.  

  ``conv_Iq(self, qgrid, mask=None, cor_factor=1)`` converts the 2D scattering data
  into a 1D scattering intensity profile.

  ``conv_Iqphi()`` and ``conv_Iqrqz()`` generate the $I(q, \phi)$ and $I(q_r, q_z)$ 
  map of the scattering data.


------------

## Axes2dPlot

  This class displays 2D scattering data in a matplotlib ``Axes``. It captures mouse
  clicks to display the reciprocal coordinates at the clicked pixel, and can overlay
  `decorations` (lines and symbols at specified reciprocal coordinates) onto the scattering data.

  ``ax, cmap`` : parameters used by matplotlib to plot the sample

  ``d2, exp`` : the scattering data and its ExpPara.

  ``mark_points()`` , ``mark_coords()`` and ``mark_standard()`` generate the decorations 
  (points, lines/grids with the given coordinates, and powder rings expected from standard
  samples) to be overlaid onto the scattering pattern. Refer to the Examples section.

  ``plot()`` plots the data. 

-----------

## Data1d

  ``trans`` is the (relative) value of the transmitted intensity, which is used as a reference
  for normlization during background subtraction. Two modes are allowed (defined as members of the 
  `Enum` `transMode` ) : `external` or `from_waxs`. For `transMode.external` , the `trans`
  value must be specified explicitly. For `transMode.from_waxs`, the `trans` value is calculated
  from the water scattering peak intensty near $2.0 \unicode{x212B}^{-1}$ .

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

  ``plot_Guinier()`` plots $ln[I(q)]$ vs $q^2$ (Guinier plot) and report $I_0$ and 
  $R_g$ .

