from py4xs.data2d import Data2d,ExpParaLiX,Axes2dPlot,DataType
from py4xs.detector_config import DetectorConfig
from py4xs.slnxs import Data1d, average
import numpy as np
import pylab as plt
import matplotlib as mpl

es = ExpParaLiX(1043, 981) 
ew1 = ExpParaLiX(619, 487) 
ew2 = ExpParaLiX(487, 619)

ene = 10790.6
wl = 2.*np.pi*1973/ene

es.wavelength = wl
es.bm_ctr_x = 374
es.bm_ctr_y = 773
es.ratioDw = 22.
es.det_orient = 0
es.det_tilt = 0
es.det_phi = 0
es.grazing_incident = False
es.flip = 1
es.incident_angle = 0.2
es.sample_normal = 0

es.calc_rot_matrix()
es.init_coordinates()

ew1.wavelength = wl
ew1.bm_ctr_x = -141    # 745
ew1.bm_ctr_y = 328.3   # 244.4
ew1.ratioDw = 2.86
ew1.det_orient = 0
ew1.det_tilt = -26
ew1.det_phi = 0
ew1.grazing_incident = False
ew1.flip = 1
ew1.incident_angle = 0.2
ew1.sample_normal = 0

ew1.calc_rot_matrix()
ew1.init_coordinates()

ew2.wavelength = wl
ew2.bm_ctr_x = 635   # 648
ew2.bm_ctr_y = 475.5   # 537.8
ew2.ratioDw = 11.32
ew2.det_orient = 0.
ew2.det_tilt = 24.
ew2.det_phi = 0.
ew2.grazing_incident = False
ew2.flip = 0
ew2.incident_angle = 0.2
ew2.sample_normal = 0

ew2.calc_rot_matrix()
ew2.init_coordinates()

es.mask.read_file("Sol-mask.SAXS")
ew1.mask.read_file("Sol-mask.WAXS1")
ew2.mask.read_file("Sol-mask.WAXS2")

qgrid = np.hstack((np.arange(0.005, 0.0499, 0.001),
                   np.arange(0.05, 0.0999, 0.002),
                   np.arange(0.1, 0.4999, 0.005),
                   np.arange(0.5, 0.9999, 0.01),
                   np.arange(1.0, 2.6,0.03)))

det_saxs = DetectorConfig(extension="_SAXS.cbf", fix_scale = 1, exp_para=es, qgrid = qgrid)
det_waxs2 = DetectorConfig(extension="_WAXS2.cbf", fix_scale = (es.Dd/ew2.Dd)**2, exp_para=ew2, qgrid = qgrid)
det_waxs1 = DetectorConfig(extension="_WAXS1.cbf", fix_scale = (es.Dd/ew1.Dd)**2, exp_para=ew1, qgrid = qgrid)

detectors = [det_saxs, det_waxs2, det_waxs1]

import os

def get_files_from_fn(fn):
    fl =[]
    fn_dir, fn_root = os.path.split(fn)
    for x in os.listdir(fn_dir):
        if (fn_root+"_0" in x) and (det_saxs.extension in x):
            fl.append(fn_dir+'/'+x.split(det_saxs.extension)[0])
    return fl
