#

## ExpPara

  Experimental parameters necessary to define the scattering geometry for 
  a 2D detector. 

  ``bm_ctr_x, bm_ctr_y``  specify the beam center position, defined as the pixel coordinates 
  of the intersection between the beam and the detector.

  ``ratioDw`` is the sample-to-detector distance, defined as a dimensionless ratio between 
  this distance and the horizontal dimenson of the detector sensing area.

  ``wavelength`` is the x-ray wavelength, in angstrom ( $\unicode{x212B}$ ) .

  ``flip`` defines how the detector image should be reoriented when read from the data file.
  The numberical value is the number of 90-deg rotations. The image is first mirrored 
  (lef and right) if the value is negative. 

  ``rot_matrix`` is the rotation matrix that translate the detector pixel position into 
  coordinates in the lab reference frame. This matrix should be calculated in classes 
  inhereted from ExpPara, where the method of calculatiing this rotation matrix is defined.

  ``sample_normal, incident_angle`` are parameters that define the sample orientation in grazing 
  incident geometry ( only effective when ``grazing_incidence`` is `True` ). 

  From these parameters, these reciprocal coordinates for each pixel are then calculated:

  ``Q`` : the amplitude of the scattering vector, in $\unicode{x212B}^{-1}$

  ``Phi`` : the azimuthal angle, $\phi$, in degrees.

  ``Qr, Qn`` : projection of the scattering vector perpendicular/paralell to the sample
  normal (grazing incidence)

  ``xQ, yQ``: projection of the scattering vector in the horizontal/vertical directions.

  These common correction factors are also calculated:

  ``FPol`` is the polarization vector, assuming that the x-ray is linearly polarized along
  the horizontal direction (synchrotron radiation).

  ``FSA``  accounts for the difference in the solid angle each pixel opens to the sample.

---------------

## Mask
  Mask for the scattering data to eliminate pixels to be excluded from data 
  processing. A mask in py4xs is typically defined as a collection of geometric
  shapes. The mask should have the same dimension as the detector image.

  ``map`` : the actual bit map that indicate excluded pixels

  ``read_file()`` : read the mask from a text file that defines the collection
  of shapes, one per line . Valid shapes include:
  
  *rectangles: r x~c~ y~c~ width height tilt_angle  
  *circles:  c x~c~ y~c~ radius                   
  *holes (inverse of circles):   h x~c~ y~c~ radius                   
  *polygons: p x~1~ y~1~ x~2~ y~2~ ... x~1~ y~1~   

  ``read_from_8bit_tif()`` : create the mask map from a 8-bit tif file. All the pixels 
  with the numerical value of 255 will be considered as part of the mask. This is useful 
  to be used together with ``Data2d.save_as_8bit_tif()``, which creates a tif file from 
  the 2D data. The tif file can be then modified in an image editor like gimp, where 
  the mask is painted white.  

  ``reload()`` : re-read the text mask that defines the mask. Any addition to the 
  mask after the last read will be lost.

  ``add_item()`` : add new shapes into the existing mask

---------------

## DetectorConfig
  All detector-related information

  ``extension`` : detector extension/designation that appears in the data file name

  ``exp_para`` : experimental parameters for the detector, as defined above

  ``mask`` : as defined above

  ``dark, flat`` : dark field and flat field data needed to apply corrections.

  ``pre_process()`` : method for performing drak field and flat feld corrections.  

