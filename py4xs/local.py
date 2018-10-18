# definitions specific to the beamline
from py4xs.exp_para import ExpParaLiX

ExpPara = ExpParaLiX

# atrtibutes needed to recreate ExpPara
exp_attr = ['wavelength', 'bm_ctr_x', 'bm_ctr_y', 'ratioDw', 
            'det_orient', 'det_tilt', 'det_phi', 'grazing_incident', 
            'flip', 'incident_angle', 'sample_normal', 'fix_azimuthal_angles']

# these are the entries in the NSLS-II data broker for the detector images
# they should match the extensions defined in the DetectorConfig instance for each detector
# at LiX there are two different configurations, hence this is defined as a list
det_names = [{"_SAXS": "pil1M_image",
              "_WAXS1": "pilW1_image",
              "_WAXS2": "pilW2_image"}, 
             {"_SAXS": "pil1M_ext_image",
              "_WAXS1": "pilW1_ext_image",
              "_WAXS2": "pilW2_ext_image"}]