# need to have a more uniform method to exchange (pack/unpack) 1D and 2D PROCESSED data with hdf5
# type of data: Data1d, MatrixWithCoordinates (not just simple numpy arrays)
import pylab as plt
import h5py
import numpy as np
import time, datetime
import os,copy,subprocess,re
import json,pickle,fabio
import multiprocessing as mp

from py4xs.slnxs import Data1d,average,filter_by_similarity,trans_mode,estimate_scaling_factor
from py4xs.utils import common_name,max_len,Schilling_p_value
from py4xs.detector_config import create_det_from_attrs
from py4xs.local import det_names,det_model,beamline_name    # e.g. "_SAXS": "pil1M_image"
from py4xs.data2d import Data2d,Axes2dPlot,MatrixWithCoords,DataType
from py4xs.utils import run
from itertools import combinations

from scipy.linalg import svd
from scipy.interpolate import splrep,sproot,splev
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as uspline

def lsh5(hd, prefix='', top_only=False, silent=False, print_attrs=True):
    """ list the content of a HDF5 file
        
        hd: a handle returned by h5py.File()
        prefix: use to format the output when lsh5() is called recursively
        top_only: returns the names of the top-level groups
        silent: suppress printouts if True
    """
    if top_only:
        tp_grps = list(hd.keys())
        if not silent:
            print(tp_grps)
        return tp_grps
    for k in list(hd.keys()):
        print(prefix, k)
        if isinstance(hd[k], h5py.Group):
            if print_attrs:
                print(list(hd[k].attrs.items()))
            lsh5(hd[k], prefix+"=", silent=silent, print_attrs=print_attrs)

def create_linked_files(fn, fnlist):
    """ create a new file to links to data in existing files in the fn_list
        for now assume that all files have the same detector/qgrid configuration without checking
    """
    ff = h5py.File(fn, 'w')
    for s in fnlist:
        fs = h5py.File(s, "r")
        if len(ff.attrs)==0:
            for an in fs.attrs:
                ff.attrs[an] = fs.attrs[an]
            ff.flush()
        for ds in lsh5(fs, top_only=True, silent=True):
            ff[ds] = h5py.ExternalLink(s, ds)
        fs.close()
    ff.close()

def qgrid_labels(qgrid):
    dq = qgrid[1]-qgrid[0]
    gpindex = [0]
    gpvalues = [qgrid[0]]
    gplabels = []

    for i in range(1,len(qgrid)-1):
        dq1 = qgrid[i+1]-qgrid[i]
        if np.fabs(dq1-dq)/dq>0.01:
            dq = dq1
            gpindex.append(i)
            prec = int(-np.log(dq)/np.log(10))+1
            gpvalues.append(qgrid[i])
    gpindex.append(len(qgrid)-1)
    gpvalues.append(qgrid[-1])

    for v in gpvalues:
        prec = int(-np.log(v)/np.log(10))+2
        gplabels.append(f"{v:.{prec}f}".rstrip('0'))
    
    return gpindex,gpvalues,gplabels
    
def pack_d1(data, ret_trans=True):
    """ utility function to creat a list of [intensity, error] from a Data1d object 
        or from a list of Data1s objects
    """
    if isinstance(data, Data1d):
        if ret_trans:
            return np.asarray([data.data,data.err]), data.trans
        else:
            return np.asarray([data.data,data.err])
    elif isinstance(data, list):
        tvs = [d.trans for d in data]
        return np.asarray([pack_d1(d, False) for d in data]),tvs
    
def unpack_d1(data, qgrid, label, trans_value):
    """ utility function to creat a Data1d object from hdf dataset
        sepatately given data[intensity and error], qgrid, label, and trans  
        works for a dataset that include a list of 1d data as well
        transMode is set to trans_mode.external
    """
    if len(data.shape)>2:
        if np.isscalar(trans_value): # this should only happen when intentionally setting trans to 0
            trans_value = np.zeros(len(data))
        return [unpack_d1(d, qgrid, label+("f%05d" % i), t) for i,(d,t) in enumerate(zip(data,trans_value))]
    else:
        ret = Data1d()
        ret.qgrid = qgrid
        ret.data = data[0]
        ret.err = data[1]
        ret.label = label
        ret.trans = trans_value
        ret.transMode = trans_mode.external
        return ret

def merge_d1s(d1s, detectors, save_merged=False, debug=False):
    """ utility function to merge 1D data sets, using functions under slnxs 
        d1s should contain data corresponding to detectors
    """
    s0 = Data1d()
    s0.qgrid = d1s[0].qgrid
    d_tot = np.zeros(s0.qgrid.shape)
    d_max = np.zeros(s0.qgrid.shape)
    d_min = np.zeros(s0.qgrid.shape)+1.e32
    e_tot = np.zeros(s0.qgrid.shape)
    c_tot = np.zeros(s0.qgrid.shape)
    label = None
    comments = ""
                
    for d1 in d1s:        
        # empty part of the data is nan
        idx = ~np.isnan(d1.data)
        d_tot[idx] += d1.data[idx]
        e_tot[idx] += d1.err[idx]
        c_tot[idx] += 1

        idx1 = (np.ma.fix_invalid(d1.data, fill_value=-1)>d_max).data
        d_max[idx1] = d1.data[idx1]
        idx2 = (np.ma.fix_invalid(d1.data, fill_value=1e32)<d_min).data
        d_min[idx2] = d1.data[idx2]
            
        comments += d1.comments
        if label is None:
            label = d1.label
        else:
            label = common_name(label, d1.label)
        
    s0.data = d_tot
    s0.err = e_tot
    idx = (c_tot>1)
    s0.overlaps.append({'q_overlap': s0.qgrid[idx],
                        'raw_data1': d_max[idx],
                        'raw_data2': d_min[idx]})
    s0.data[idx] /= c_tot[idx]
    s0.err[idx] /= np.sqrt(c_tot[idx])
    s0.label = label
    s0.comments = comments # .replace("# ", "## ")
    if save_merged:
        s0.save(s0.label+".dd", debug=debug)
        
    return s0


# copied from pipeline-test: merge, fix_angular_range, interp_d2
def merge(ds):
    """ merge a list of MatrixWithCoord together
        the datatype should be DataType.qphi
    """
    if len(ds)==1:
        return ds[0].copy()
    
    wt = np.zeros(ds[0].shape)
    avg = np.zeros(ds[0].shape)
    idx = None
    for d in ds:
        if d.shape!=avg.shape:
            raise Exception("merge: the two data sets must have the same shape: ", d.shape, avg.shape)
        idx = ~np.isnan(d)
        avg[idx] += d[idx]
        wt[idx] += 1

    idx = (wt>0)
    avg[idx] /= wt[idx]
    avg[~idx] = np.nan
    
    return avg
    
def fix_angular_range(da):
    """ da should be a numpy array
        return modified angular range between -180 and 180
        assume that the angular value is not too far off to begin with
    """
    da1 = np.copy(da)
    da1[da1>180] -= 360    # worse case some angles may go up to 360+delta 
    da1[da1<-180] += 360   # this shouldn't happen
    return da1

def interp_d2(d2, method="spline", param=0.05):
    """ d2 is a 2d array
        interpolate within each row 
        methods should be "linear" or "spline"
        
        a better version of this should use 2d interpolation
        but only fill in the space that is narrow enough in one direction (e.g. <5 missing data points)
        
    """
    h,w = d2.shape
    
    xx1 = np.arange(w)
    for k in range(h):
        yy1 = d2[k,:]    
        idx = ~np.isnan(yy1)
        if len(idx)<=10:  # too few valid data points
            continue
        idx1 = np.where(idx)[0]
        # only need to refill the values that are currently nan
        idx2 = np.copy(idx)
        idx2[:idx1[0]] = True
        idx2[idx1[-1]:] = True
        if method=="linear":
            d2[k,~idx2] = np.interp(xx1[~idx2], xx1[idx], yy1[idx])
        elif method=="spline":
            fs = uspline(xx1[idx], yy1[idx])
            fs.set_smoothing_factor(param)
            d2[k,~idx2] = fs(xx1[~idx2])
        else:
            raise Exception(f"unknown method for intepolation: {method}")


def proc_2d(queue, images, sn, nframes, detectors, qphi_range, debug, starting_frame_no=0):
    """ convert 2D data to q-phi map
        may want to do this separately for SAXS and WAXS; how to specify?
    """
    pass

def proc_make_thumnails(queue, images, sn, nframes, detectors, qphi_range, debug, starting_frame_no=0):
    """ make thumbnails, specify the detector, output dataset name, color scale, etc.
    """
    pass

def proc_line_profile(queue, images, sn, nframes, detectors, qphi_range, debug, starting_frame_no=0):
    """ put the results in a dataset, with attributes describing where the results come from?
    """
    pass

def proc_d1merge(args):
    """ utility function to perfrom azimuthal average and merge detectors
    """
    images,sn,nframes,starting_frame_no,debug,detectors,qgrid,reft,save_1d,save_merged = args
    ret = {'merged': []}
    sc = {}
    
    for det in detectors:
        ret[det.extension] = []
        if det.fix_scale is not None:
            sc[det.extension] = 1./det.fix_scale

    if debug is True:
        print("processing started: sample = %s, starting frame = #%d" % (sn, starting_frame_no))
    for i in range(nframes):
        for det in detectors:
            dt = Data1d()
            label = "%s_f%05d%s" % (sn, i+starting_frame_no, det.extension)
            dt.load_from_2D(images[det.extension][i], 
                            det.exp_para, qgrid, det.pre_process, det.exp_para.mask,
                            save_ave=False, debug=debug, label=label)
            dt.scale(sc[det.extension])
            ret[det.extension].append(dt)
    
        dm = merge_d1s([ret[det.extension][i] for det in detectors], detectors, save_merged, debug)
        ret['merged'].append(dm)
            
    if debug is True:
        print("processing completed: ", sn, starting_frame_no)

    return [sn, starting_frame_no, ret]
        
def proc_sample(queue, images, sn, nframes, detectors, qgrid, reft, save_1d, save_merged, debug,
               starting_frame_no=0, transMode=None, monitor_counts=None):
    """ utility function to perfrom azimuthal average and merge detectors
    """
    ret = {'merged': []}
    sc = {}
    
    for det in detectors:
        ret[det.extension] = []
        if det.fix_scale is not None:
            sc[det.extension] = 1./det.fix_scale

    if debug is True:
        print("processing started: sample = %s, starting frame = #%d" % (sn, starting_frame_no))
    for i in range(nframes):
        for det in detectors:
            dt = Data1d()
            label = "%s_f%05d%s" % (sn, i+starting_frame_no, det.extension)
            dt.load_from_2D(images[det.extension][i+starting_frame_no], 
                            det.exp_para, qgrid, det.pre_process, det.exp_para.mask,
                            save_ave=False, debug=debug, label=label)
            dt.scale(sc[det.extension])
            ret[det.extension].append(dt)
    
        dm = merge_d1s([ret[det.extension][i] for det in detectors], detectors, save_merged, debug)
        ret['merged'].append(dm)
            
    if debug is True:
        print("processing completed: ", sn, starting_frame_no)
    if queue is None: # single-thread
        return ([sn,starting_frame_no,ret])
    else: # multi-processing    
        queue.put([sn,starting_frame_no,ret])

class h5exp():
    """ empty h5 file for exchanging exp_setup/qgrid
    """
    def __init__(self, fn, exp_setup=None):
        self.fn = fn
        if exp_setup==None:     # assume the h5 file will provide the detector config
            self.qgrid = self.read_detectors()
        else:
            self.detectors, self.qgrid = exp_setup
            self.save_detectors()
        
    def save_detectors(self):
        self.fh5 = h5py.File(self.fn, "w")   # new file
        dets_attr = [det.pack_dict() for det in self.detectors]
        self.fh5.attrs['detectors'] = json.dumps(dets_attr)
        self.fh5.attrs['qgrid'] = list(self.qgrid)
        self.fh5.flush()
        self.fh5.close()
    
    def read_detectors(self):
        self.fh5 = h5py.File(self.fn, "r")   # file must exist
        dets_attr = self.fh5.attrs['detectors']
        qgrid = self.fh5.attrs['qgrid']
        self.detectors = [create_det_from_attrs(attrs) for attrs in json.loads(dets_attr)]  
        self.fh5.close()
        return np.asarray(qgrid)
    
    def recalibrate(self, fn_std, energy=-1, 
                    det_type={"_SAXS": "Pilatus1M", "_WAXS2": "Pilatus1M"},
                    bkg={}):
        """ fn_std should be a h5 file that contains AgBH pattern
            use the specified energy (keV) if the value is valid
            detector type
        """
        pxsize = 0.172e-3
        dstd = h5xs(fn_std, [self.detectors, self.qgrid]) 
        uname = os.getenv("USER")
        sn = dstd.samples[0]
        if energy>5. and energy<20.:
            wl = 2.*np.pi*1.973/energy
            for det in self.detectors:
                det.exp_para.wavelength = wl
        for det in self.detectors:
            print(f"processing detector {det.extension} ...")    
            ep = det.exp_para
            poni_file = f"/tmp/{uname}{det.extension}.poni"
            data_file = f"/tmp/{uname}{det.extension}.cbf"
            img = dstd.fh5["%s/primary/data/%s" % (sn, dstd.det_name[det.extension])][0]

            # this would work better if the detector geometry specification 
            # can be more flexible for pyFAI-recalib 
            if ep.flip: ## can only handle flip=1 right now
                if ep.flip!=1: 
                    raise Exception(f"don't know how to handle flip={ep.flip}.")
                poni1 = pxsize*ep.bm_ctr_x
                poni2 = pxsize*(ep.ImageHeight-ep.bm_ctr_y)
                dmask = np.fliplr(det.exp_para.mask.map.T)
            else: 
                poni1 = pxsize*ep.bm_ctr_y
                poni2 = pxsize*ep.bm_ctr_x
                dmask = det.exp_para.mask.map
            fabio.cbfimage.CbfImage(data=img*(~dmask)).write(data_file)

            poni_file_text = ["poni_version: 2",
                              f"Detector: {det_type[det.extension]}",
                              "Detector_config: {}",
                              f"Distance: {pxsize*ep.Dd}",
                              f"Poni1: {poni1}", # y-axis
                              f"Poni2: {poni2}", # x-axis
                              "Rot1: 0.0", "Rot2: 0.0", "Rot3: 0.0",
                              f"Wavelength: {ep.wavelength*1e-10:.4g}"]
            fh = open(poni_file, "w")
            fh.write("\n".join(poni_file_text))
            fh.close()
            if det.extension in bkg.keys():
                cmd = ["pyFAI-recalib", "-i", poni_file, "-b", f"{bkg[det.extension]}", 
                       "-c", "AgBh", "-r", "11", "--no-tilt", "--no-gui", "--no-interactive", data_file]
            else:
                cmd = ["pyFAI-recalib", "-i", poni_file, 
                       "-c", "AgBh", "-r", "11", "--no-tilt", "--no-gui", "--no-interactive", data_file]
            print(" ".join(cmd))
            ret = run(cmd)
            txt = ret.strip().split('\n')[-1]
            #print(txt)
            print(f"  Original ::: bm_ctr_x = {ep.bm_ctr_x:.2f}, bm_ctr_y = {ep.bm_ctr_y:.2f}, ratioDw = {ep.ratioDw:.3f}")
            d,xc,yc = np.asarray(re.findall('\d+\.\d*', txt), dtype=np.float)[:3]
            dr = d/(ep.Dd*pxsize)/1000  # d is in mm
            ep.ratioDw *= dr
            if ep.flip: ## can only handle flip=1 right now
                ep.bm_ctr_x = yc
                ep.bm_ctr_y = ep.ImageHeight-xc
            else: 
                ep.bm_ctr_y = yc
                ep.bm_ctr_x = xc
            print(f"   Revised ::: bm_ctr_x = {ep.bm_ctr_x:.2f}, bm_ctr_y = {ep.bm_ctr_y:.2f}, ratioDw = {ep.ratioDw:.3f}")
            ep.init_coordinates()
        self.save_detectors()
        
class h5xs():
    """ Scattering data in transmission geometry
        Transmitted beam intensity can be set either from the water peak (sol), or from intensity monitor.
        Data processing can be done either in series, or in parallel. Serial processing can be forced.
        
    """    
    def __init__(self, fn, exp_setup=None, transField='', save_d1=True):
        """ exp_setup: [detectors, qgrid]
            transField: the intensity monitor field packed by suitcase from databroker
            save_d1: save newly processed 1d data back to the h5 file
        """
        self.d1s = {}
        self.detectors = None
        self.samples = []
        self.attrs = {}
        # name of the dataset that contains transmitted beam intensity, e.g. em2_current1_mean_value
        self.transField = None  
        self.transStream = None  

        self.fn = fn
        self.save_d1 = save_d1
        self.fh5 = h5py.File(self.fn, "r+")   # file must exist
        if exp_setup==None:     # assume the h5 file will provide the detector config
            self.qgrid = self.read_detectors()
        else:
            self.detectors, self.qgrid = exp_setup
            self.save_detectors()
        self.list_samples(quiet=True)

        # find out what are the fields corresponding to the 2D detectors
        # at LiX there are two possibilities
        sn = self.samples[0]
        streams = list(self.fh5[f"{sn}"])
        data_fields = {}
        for stnm in streams:
            if 'data' in self.fh5[f"{sn}/{stnm}"].keys(): 
                for tf in list(self.fh5[f"{sn}/{stnm}/data"]):
                    data_fields[tf] = stnm
        
        self.det_name = None
        # these are the detectors that are present in the data
        d_dn = [d.extension for d in self.detectors]
        for det_name in det_names:
            for k in set(det_name.keys()).difference(d_dn):
                del det_name[k]
            if set(det_name.values()).issubset(data_fields.keys()):
                self.det_name = det_name
                break
        if self.det_name is None:
            print('fields in the h5 file: ', data_fields)
            raise Exception("Could not find the data corresponding to the detectors.")
        if transField=='': 
            if 'trans' in self.fh5.attrs:
                tf = self.fh5.attrs['trans'].split(',')
                if len(tf)<3:  # in earlier code, transStream is not recorded
                    [v, self.transField] = tf
                else: 
                    [v, self.transField, self.transStream] = tf
                self.transMode = trans_mode(int(v))
                return
            else:
                self.transMode = trans_mode.from_waxs
                self.transField = ''
                self.transStream = ''
        elif transField not in list(data_fields.keys()):
            print("invalid field for transmitted intensity: ", transField)
            raise Exception()
        else:
            self.transField = transField
            self.transStream = data_fields[transField]
            self.transMode = trans_mode.external
        self.fh5.attrs['trans'] = ','.join([str(self.transMode.value), self.transField, self.transStream])
        self.fh5.flush()
            
    def save_detectors(self):
        dets_attr = [det.pack_dict() for det in self.detectors]
        self.fh5.attrs['detectors'] = json.dumps(dets_attr)
        self.fh5.attrs['qgrid'] = list(self.qgrid)
        self.fh5.flush()
    
    def read_detectors(self):
        dets_attr = self.fh5.attrs['detectors']
        qgrid = self.fh5.attrs['qgrid']
        self.detectors = [create_det_from_attrs(attrs) for attrs in json.loads(dets_attr)]  
        return np.asarray(qgrid)

    def md_dict(self, sn, md_keys=[]):
        """ create the meta data to be recorded in ascii data files
            from the detector_config (attribute of the h5xs object) 
            and scan header (in the dataset attribute, value set when writting h5 using suitcase)
            based on SASDBD

            example for static samples:

            Instrument: BM29
            Detector: Pilatus 2M
            Date: 2017-10-17
            Wavelength (nm): 0.1240
            (Or, alternatively: X-ray energy (keV): 10.00)
            Sample-to-detector distance (m): 2.01
            Exposure time/frame (s): 0.100
            Sample temperature (C): 20.0

            example for inline SEC:

            instrument: BM29
            Detector: Pilatus 2M
            Column type: S75 Increase
            Date: 2017-10-17
            Wavelength (nm): 0.1240
            (Or, alternatively: X-ray energy (keV): 10.00)
            Sample-to-detector distance (m): 2.01
            Exposure time/frame (s): 0.995
            Sample temperature (C): 20.0
            Flow rate (ml/min): 0.500
            Sample injection concentration (mg/ml): 25.0
            Sample injection volume (ml): 0.0750
        """    
        md = {}
        bshdr = json.loads(self.fh5[sn].attrs['start'])
        md["Instrument"] = bshdr['beamline_id']
        ts = time.localtime(bshdr['time'])
        md["Date"] = time.strftime("%Y-%m-%d", ts)
        md["Time"] = time.strftime("%H:%M:%S", ts)
        ene = bshdr["energy"]["energy"]
        md["Wavelength (A)"] = f"{2.*3.1416*1973/ene:.4f}"
        
        try:
            bscfg = json.loads(self.fh5[sn].attrs["descriptors"])[0]['configuration']
            for det in self.detectors:
                dn = self.det_name[det.extension].strip("_image")
                exp = bscfg[dn]['data'][f"{dn}_cam_acquire_time"]
                if not "Detector" in md.keys():
                    md["Detector"] = det_model[det.extension]
                    md["Exposure time/frame (s)"] = f"{exp:.3f}"
                    md["Sample-to-detector distance (m): "] = f"{det.s2d_distance/1000: .3f}"
                else:
                    md["Detector"] += f" , {det_model[det.extension]}"
                    md["Exposure time/frame (s)"] += f" , {exp:.3f}" 
                    md["Sample-to-detector distance (m): "] += f" , {det.s2d_distance/1000: .3f}"
        except:  # the header information may be incomplete
            pass
                    
        for k in md_keys:
            if k in bshdr.keys():
                md[k] = bshdr[k]

        return md
                
    def md_string(self, sn, md_keys=[]):

        md = self.md_dict(sn, md_keys=md_keys)
        md_str = ""
        for k in md.keys():
            md_str += f"# {k} : {md[k]}\n"
        
        return md_str
    
    def list_samples(self, quiet=False):
        self.samples = lsh5(self.fh5, top_only=True, silent=True)
        if not quiet:
            print(self.samples)
    
    def get_d2(self, sn=None, det_ext=None, frn=0):
        if sn is None:
            sn = self.samples[0]
        d2s = {}
        for det in self.detectors:
            dset = self.fh5["%s/primary/data/%s" % (sn, self.det_name[det.extension])]
            if len(dset.shape)>3:  # need to make more effort
                dset = dset[0]
            d2 = Data2d(dset[frn], exp=det.exp_para)
            d2s[det.extension] = d2
        if not det_ext:
            return d2s
        else:
            return d2s[det_ext]
                
    def check_bm_center(self, sn=None, det_ext='_SAXS', frn=0, 
                        qs=0.005, qe=0.05, qn=100, Ns=9):
        """ this function compares the beam intensity on both sides of the beam center,
            and advise if the beam center as defined in the detector configuration is incorrect
            dividing data into 4*Ns slices, show data in horizontal and vertical cuts
        """
        i = 0
        d2 = self.get_d2(sn=sn, det_ext=det_ext, frn=frn)
        qg = np.linspace(qs, qe, qn)
        d2.conv_Iqphi(Nq=qg, Nphi=Ns*4, mask=d2.exp.mask)

        dch1 = d2.qphi_data.d[i]
        dch2 = d2.qphi_data.d[i+2*Ns]
        dcv1 = d2.qphi_data.d[i+Ns]
        dcv2 = d2.qphi_data.d[i+3*Ns]

        sh0,dh0 = max_len(dch1, dch2, return_all=True)
        ph0 = Schilling_p_value(qn, np.max(dh0))
        sh1,dh1 = max_len(dch1[1:], dch2[:-1], return_all=True)
        ph1 = Schilling_p_value(qn, np.max(dh1))
        sh_1,dh_1 = max_len(dch1[:-1], dch2[1:], return_all=True)
        ph_1 = Schilling_p_value(qn, np.max(dh_1))

        print(f"horizontal cuts:")
        print(f"p-values(C): {ph0:.4f} (as is), {ph1:.4f} (shift +1), {ph_1:.4f} (shift -1)")
        fig = plt.figure(figsize=(6,3), constrained_layout=True)
        #fig.subplots_adjust(bottom=0.3)
        gs = fig.add_gridspec(2,2)

        fig.add_subplot(gs[:, 0])
        plt.semilogy(d2.qphi_data.xc, dch1)
        plt.semilogy(d2.qphi_data.xc, dch2)        
        
        fig.add_subplot(gs[1, 1])
        plt.bar(range(len(sh0)), (sh0*2-1)/3, bottom=0, width=1)
        plt.bar(range(len(sh1)), (sh1*2-1)/3, bottom=1, width=1)
        plt.bar(range(len(sh_1)), (sh_1*2-1)/3, bottom=-1, width=1)
        plt.xlabel("point position")

        fig.add_subplot(gs[0, 1])
        maxb = 2.5+int(np.max(np.hstack((dh0,dh1,dh_1,[qn/3]))))
        plt.hist(dh0, bins=np.arange(0.5, maxb, 1), bottom=0, density=True)
        plt.hist(dh1, bins=np.arange(0.5, maxb, 1), bottom=1, density=True)
        plt.hist(dh_1, bins=np.arange(0.5, maxb, 1), bottom=-1, density=True)
        plt.ylim(-1.2,1.8)
        plt.xlabel("patch size")
        
        sv0,dv0 = max_len(dcv1, dcv2, return_all=True)
        pv0 = Schilling_p_value(qn, np.max(dv0))
        sv1,dv1 = max_len(dcv1[1:], dcv2[:-1], return_all=True)
        pv1 = Schilling_p_value(qn, np.max(dv1))
        sv_1,dv_1 = max_len(dcv1[:-1], dcv2[1:], return_all=True)
        pv_1 = Schilling_p_value(qn, np.max(dv_1))

        print(f"vertical cuts:")
        print(f"p-values(C): {pv0:.4f} (as is), {pv1:.4f} (shift +1), {pv_1:.4f} (shift -1)")
        fig = plt.figure(figsize=(6,3), constrained_layout=True)
        #fig.subplots_adjust(bottom=0.3)
        gs = fig.add_gridspec(2,2)

        fig.add_subplot(gs[:, 0])
        plt.semilogy(d2.qphi_data.xc, dcv1)
        plt.semilogy(d2.qphi_data.xc, dcv2)        
        
        fig.add_subplot(gs[1, 1])
        plt.bar(range(len(sv0)), (sv0*2-1)/3, bottom=0, width=1)
        plt.bar(range(len(sv1)), (sv1*2-1)/3, bottom=1, width=1)
        plt.bar(range(len(sv_1)), (sv_1*2-1)/3, bottom=-1, width=1)
        plt.xlabel("point position")

        fig.add_subplot(gs[0, 1])
        maxb = 2.5+int(np.max(np.hstack((dv0,dv1,dv_1,[qn/3]))))
        plt.hist(dv0, bins=np.arange(0.5, maxb, 1), bottom=0, density=True)
        plt.hist(dv1, bins=np.arange(0.5, maxb, 1), bottom=1, density=True)
        plt.hist(dv_1, bins=np.arange(0.5, maxb, 1), bottom=-1, density=True)
        plt.ylim(-1.2,1.8)
        plt.xlabel("patch size")
        
        dq = d2.qphi_data.xc[1]-d2.qphi_data.xc[0]
        X,Y = d2.exp.calc_from_QPhi(np.asarray([0., dq]), np.asarray([0., 0.]))
        dx = np.sqrt((X[1]-X[0])**2+(Y[1]-Y[0])**2)
        print(f"1 data point = {dx:.2f} pixels")
        plt.show()
        
    def show_data0(self, sn=None, det_ext='_SAXS', frn=0, ax=None,
                  logScale=True, showMask=False, clim=(0.1,14000), showRef=True, cmap=None):
        """ display frame #frn of the data under det for sample sn
        """
        d2 = self.get_d2(sn=sn, det_ext=det_ext, frn=frn)
        if ax is None:
            plt.figure()
            ax = plt.gca()
        pax = Axes2dPlot(ax, d2.data, exp=d2.exp)
        pax.plot(log=logScale)
        if cmap is not None:
            pax.set_color_scale(plt.get_cmap(cmap)) 
        if showMask:
            pax.plot(log=logScale, mask=d2.exp.mask)
        pax.img.set_clim(*clim)
        pax.coordinate_translation="xy2qphi"
        if showRef:
            pax.mark_standard("AgBH", "r:")
        pax.capture_mouse()

    def show_data(self, sn=None, det_ext='_SAXS', frn=0, ax=None,
                  logScale=True, showMask=False, clim=(0.1,14000), showRef=True, cmap=None):
        """ display frame #frn of the data under det for sample sn
        """
        d2 = self.get_d2(sn=sn, det_ext=det_ext, frn=frn)
        if ax is None:
            plt.figure()
            ax = plt.gca()
        
        pax = Axes2dPlot(ax, d2.data, exp=d2.exp)
        pax.plot(log=logScale)
        if cmap is not None:
            pax.set_color_scale(plt.get_cmap(cmap)) 
        if showMask:
            pax.plot(log=logScale, mask=d2.exp.mask)
        pax.img.set_clim(*clim)
        pax.coordinate_translation="xy2qphi"
        if showRef:
            pax.mark_standard("AgBH", "r:")
        pax.capture_mouse()
        #plt.show() 
    
    def show_data_qxy(self, sn=None, frn=0, ax=None, dq=0.006,
                      fig_type="qxy", apply_sym=False, fix_gap=False,
                      logScale=True, showMask=False, clim=(0.1,14000), showRef=True, cmap=None):
        """ display frame #frn of the data under det for sample sn
            det is a list of detectors, or a string, data file extension
            fig_type should be either "qxy" or "qphi"
        """
        d2s = self.get_d2(sn=sn, frn=frn)
        if ax is None:
            plt.figure()
            ax = plt.gca()

        xqmax = np.max([d.exp_para.xQ.max() for d in self.detectors])
        xqmin = np.min([d.exp_para.xQ.min() for d in self.detectors])
        yqmax = np.max([d.exp_para.yQ.max() for d in self.detectors])
        yqmin = np.min([d.exp_para.yQ.min() for d in self.detectors])

        xqmax = np.floor(xqmax/dq)*dq
        xqmin = np.ceil(xqmin/dq)*dq
        yqmax = np.floor(yqmax/dq)*dq
        yqmin = np.ceil(yqmin/dq)*dq

        xqgrid = np.arange(start=xqmin, stop=xqmax+dq, step=dq)
        yqgrid = np.arange(start=yqmin, stop=yqmax+dq, step=dq)        

        xyqmaps = []
        for dn in d2s.keys():   
            if showMask:
                mask = d2s[dn].exp.mask
            else:
                mask = None
            mp = d2s[dn].data.conv(xqgrid, yqgrid, 
                                   d2s[dn].exp.xQ, d2s[dn].exp.yQ, 
                                   mask=mask)
            mp.d *= (d2s[dn].exp.Dd/d2s["_SAXS"].exp.Dd)**2
            xyqmaps.append(mp.d)
        xyqmap = merge(xyqmaps)

        if logScale:
            plt.imshow(np.log(xyqmap), extent=(xqmin, xqmax, yqmin, yqmax))
            plt.clim(np.log(clim))
        else:
            plt.imshow(xyqmap, extent=(xqmin, xqmax, yqmin, yqmax))
            plt.clim(clim)

    def show_data_qphi(self, sn=None, frn=0, ax=None, Nq=200, Nphi=60,
                       apply_symmetry=False, fill_gap=False, interp_method='linear',
                       logScale=True, showMask=False, clim=(0.1,14000), showRef=True, cmap=None):
        d2s = self.get_d2(sn=sn, frn=frn)
        if ax is None:
            plt.figure()
            ax = plt.gca()

        qmax = np.max([d.exp_para.Q.max() for d in self.detectors]) 
        qmin = np.min([d.exp_para.Q.min() for d in self.detectors]) 
        # keep 2 significant digits only for the step_size
        dq = (qmax-qmin)/Nq
        n = int(np.floor(np.log10(dq)))
        sc = np.power(10., n)
        dq = np.around(dq/sc, 1)*sc

        qmax = dq*np.ceil(qmax/dq)
        qmin = dq*np.floor(qmin/dq)
        Nq = int((qmax-qmin)/dq+1)

        q_grid = np.linspace(qmin, qmax, Nq) 

        Nphi = 2*int(Nphi/2)+1
        phi_grid = np.linspace(-180., 180, Nphi)

        dms = []
        for dn in d2s.keys():   
            if showMask:
                mask = d2s[dn].exp.mask
            else:
                mask = None
            # since the q/phi grids are specified, use the MatrixWithCoord.conv() function
            # the grid specifies edges of the bin for histogramming
            # the phi value in exp_para may not always be in the range of (-180, 180)
            dm = d2s[dn].data.conv(q_grid, phi_grid, 
                                   d2s[dn].exp.Q, 
                                   fix_angular_range(d2s[dn].exp.Phi),
                                   mask=mask, datatype=DataType.qphi)

            d1 = dm.d*(d2s[dn].exp.Dd/d2s["_SAXS"].exp.Dd)**2
            if apply_symmetry:
                Np = int(Nphi/2)
                d2 = np.vstack([d1[Np:,:], d1[:Np,:]])
                d1 = merge([d1,d2])

            # not attempt to extrapolate into non-coveraged area in reciprocal space
            # doing interpolation here since the non-coverage area is well defined for 
            # a single detector, but may not be in merged data 
            if fill_gap:
                interp_d2(d1, method=interp_method)

            dms.append(d1)

        qphimap = merge(dms)
        if logScale:
            plt.imshow(np.log(qphimap), extent=(qmin, qmax, -180, 180), aspect="auto")
            plt.clim(np.log(clim))
        else:
            plt.imshow(qphiqmap, extent=(qmin, qmax, -180, 180), aspect="auto")
            plt.clim(clim)

    def set_trans(self, sn=None, transMode=None, interpolate=False, gf_sigma=5):
        """ set the transmission values for the merged data
            the trans values directly from the raw data (water peak intensity or monitor counts)
            but this value is changed if the data is scaled
            the trans value for all processed 1d data are saved as attrubutes of the dataset
        """
        if transMode is not None:
            self.transMode = transMode
        if self.transMode==None:
            raise Exception("a valid transmited intensity mode must be specified.")
        
        if sn is None:
            samples = self.samples
        else:
            samples = [sn]
        for s in samples:
            if self.transMode==trans_mode.external: # transField should be set/validated already
                trans_data = self.fh5[f'{s}/{self.transStream}/data/{self.transField}'][...]
                ts = self.fh5[f'{s}/{self.transStream}/timestamps/{self.transField}'][...]
                if interpolate:  ## smoothing really
                    spl = UnivariateSpline(ts, gaussian_filter(trans_data, gf_sigma))
                    ts0 = self.fh5[f'{s}/primary/timestamps/{list(self.det_name.values())[0]}'][...]
                    trans_data = spl(ts0)
        
            # these are the datasets that needs updated trans values
            if 'merged' not in self.d1s[s].keys():
                continue
            t_values = []
            for i in range(len(self.d1s[s]['merged'])):
                if self.transMode==trans_mode.external:
                    self.d1s[s]['merged'][i].set_trans(trans_data[i], transMode=self.transMode)
                else:
                    self.d1s[s]['merged'][i].set_trans(transMode=self.transMode)
                t_values.append(self.d1s[s]['merged'][i].trans)
            if 'averaged' not in self.d1s[s].keys():
                continue
            self.d1s[s]['averaged'].trans = np.average(t_values)
    
    def load_d1s(self, sn=None):
        """ load the processed 1d data saved in the hdf5 file into memory 
            for each sample
                 attribute "selected": which raw data are included in average
                 attribute "sc_factor": scaling factor used for buffer subtraction
            raw data: from each detector, and 'merged'
            averaged data: averaged from multiple frames
            corrected data: subtracted for buffer scattering
            buffer data will be recreated based on buffer assignment 
        """        
        if sn==None:
            self.list_samples(quiet=True)
            for sn in self.samples:
                self.load_d1s(sn)
        
        fh5 = self.fh5
        if "processed" not in lsh5(fh5[sn], top_only=True, silent=True): 
            return
        
        if sn not in list(self.attrs.keys()):
            self.d1s[sn] = {}
            self.attrs[sn] = {}
        grp = fh5[sn+'/processed']
        for k in list(grp.attrs.keys()):
            self.attrs[sn][k] = grp.attrs[k]   
        for k in lsh5(grp, top_only=True, silent=True):
            if 'trans' in grp[k].attrs.keys():
                tvs = grp[k].attrs['trans']
            else:
                tvs = 0
            self.d1s[sn][k] = unpack_d1(grp[k], self.qgrid, sn+k, tvs)   

    def save_d1s(self, sn=None, debug=False):
        """
        save the 1d data in memory to the hdf5 file 
        processed data go under the group sample_name/processed
        assume that the shape of the data is unchanged
        """
        
        if self.save_d1 is False:
            print("requested to save_d1s() but h5xs.save_d1 is False.")
            return
        if sn==None:
            self.list_samples(quiet=True)
            for sn in self.samples:
                self.save_d1s(sn)
        
        fh5 = self.fh5        
        if "processed" not in list(lsh5(fh5[sn], top_only=True, silent=True)):
            grp = fh5[sn].create_group("processed")
        else:
            grp = fh5[sn+'/processed']
            g0 = lsh5(grp, top_only=True, silent=True)[0]
            if grp[g0][0].shape[1]!=len(self.qgrid): # if grp[g0].value[0].shape[1]!=len(self.qgrid):
                # new size for the data
                del fh5[sn+'/processed']
                grp = fh5[sn].create_group("processed")
        
        # these attributes are not necessarily available when save_d1s() is called
        if sn in list(self.attrs.keys()):
            for k in list(self.attrs[sn].keys()):
                grp.attrs[k] = self.attrs[sn][k]
                if debug is True:
                    print("writting attribute to %s: %s" % (sn, k))

        ds_names = lsh5(grp, top_only=True, silent=True)
        for k in list(self.d1s[sn].keys()):
            data,tvs = pack_d1(self.d1s[sn][k])
            if debug is True:
                print("writting attribute to %s: %s" % (sn, k))
            if k not in ds_names:
                grp.create_dataset(k, data=data)
            else:
                grp[k][...] = data   

            # save trans values for processed data
            # before 1d data merge, the trans value should be 0              
            # on the other hand there could be data collected with the beam off, therefore trans=0
            if (np.asarray(tvs)>0).any(): 
                grp[k].attrs['trans'] = tvs
                
        fh5.flush()

    def average_d1s(self, samples=None, update_only=False, selection=None, filter_data=False, debug=False):
        """ if update_only is true: only work on samples that do not have "merged' data
            selection: if None, retrieve from dataset attribute
        """
        if debug is True:
            print("start processing: average_samples()")
            t1 = time.time()

        if samples is None:
            samples = self.samples
        elif isinstance(samples, str):
            samples = [samples]
        for sn in samples:
            if update_only and 'merged' in list(self.d1s[sn].keys()): continue
                
            if filter_data:
                d1keep,d1disc = filter_by_similarity(self.d1s[sn]['merged'])
                self.attrs[sn]['selected'] = []
                for d1 in self.d1s[sn]['merged']:
                    self.attrs[sn]['selected'].append(d1 in d1keep)
            else:
                if  selection==None:
                    # if there is an error, run with filter_data=True
                    selection = self.attrs[sn]['selected']
                else:
                    self.attrs[sn]['selected'] = selection
                d1keep = [self.d1s[sn]['merged'][i] for i in range(len(selection)) if selection[i]]
            
            if len(d1keep)>1:
                self.d1s[sn]['averaged'] = d1keep[0].avg(d1keep[1:], debug=debug)
            else:
                self.d1s[sn]['averaged'] = d1keep[0]
                
        self.save_d1s(debug=debug)

        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    def plot_d1s(self, sn, ax=None, offset=1.5, fontsize="large",
                    show_overlap=False, show_subtracted=False, show_subtraction=True):
        """ show_subtracted:
                work only if sample is background-subtracted, show the subtracted result
            show_subtraction: 
                if True, show sample and boffer when show_subtracted
            show_overlap: 
                also show data in the overlapping range from individual detectors
                only allow if show_subtracted is False
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()
        if show_subtracted:
            if 'subtracted' not in list(self.d1s[sn].keys()):
                raise Exception("bkg-subtracted data not found: ", sn)
            self.d1s[sn]['subtracted'].plot(ax=ax, fontsize=fontsize)
            if show_subtraction:
                self.d1s[sn]['averaged'].plot(ax=ax, fontsize=fontsize)
                self.d1b[sn].plot(ax=ax, fontsize=fontsize)
        else:
            sc = 1
            for i,d1 in enumerate(self.d1s[sn]['merged']):
                if self.attrs[sn]['selected'][i]:
                    d1.plot(ax=ax, scale=sc, fontsize=fontsize)
                    ax.plot(self.d1s[sn]['averaged'].qgrid, 
                            self.d1s[sn]['averaged'].data*sc, 
                            color="gray", lw=2, ls="--")
                    if show_overlap:
                        for det1,det2 in combinations(list(self.det_name.keys()), 2):
                            idx_ov = ~np.isnan(self.d1s[sn][det1][i].data) & ~np.isnan(self.d1s[sn][det2][i].data) 
                            if len(idx_ov)>0:
                                ax.plot(self.d1s[sn][det1][i].qgrid[idx_ov], 
                                        self.d1s[sn][det1][i].data[idx_ov]*sc, "y^")
                                ax.plot(self.d1s[sn][det2][i].qgrid[idx_ov], 
                                        self.d1s[sn][det2][i].data[idx_ov]*sc, "gv")
                else:
                    ax.plot(self.d1s[sn]['merged'][i].qgrid, self.d1s[sn]['merged'][i].data*sc, 
                            color="gray", lw=2, ls=":")
                sc *= offset
                
    def compare_d1s(self, samples=None, show_subtracted=True, ax=None):
        if samples is None:
            samples = self.samples
        if show_subtracted:
            grp = "subtracted"
        else: 
            grp = "averaged"

        sl = []  
        for s in samples:
            if "averaged" not in self.d1s[s].keys():
                raise Exception(f"processed data not found for {s}.")
            if grp in self.d1s[s].keys():
                sl.append(s)
        if len(sl)==0:
            print("no suitable data to plot ...")
            return    
        
        if ax is None:
            plt.figure()
            ax = plt.gca()
        
        for s in sl:
            self.d1s[s][grp].plot(ax=ax)
        

    def export_d1s(self, samples=None, path="", save_subtracted=True, debug=False):
        """ if path is used, be sure that it ends with '/'
        """
        if samples is None:
            samples = self.samples
            #samples = list(self.buffer_list.keys())
        elif isinstance(samples, str):
            samples = [samples]

        if debug is True:
            print("start processing: export_d1s()")
        for sn in samples:
            if debug is True:
                print(f"   processing: {sn} ...")
            if save_subtracted=="merged":
                if 'merged' not in self.d1s[sn].keys():
                    print("1d data not available.")
                    continue
                for i in range(len(self.d1s[sn]['merged'])):
                    self.d1s[sn]['merged'][i].save("%s%s_%d%c.dat"%(path,sn,i,'m'), debug=debug, 
                                                footer=self.md_string(sn))                                    
            elif save_subtracted is True:
                if 'subtracted' not in self.d1s[sn].keys():
                    print("subtracted data not available.")
                    continue
                self.d1s[sn]['subtracted'].save("%s%s_%c.dat"%(path,sn,'s'), debug=debug, 
                                                footer=self.md_string(sn))
            else: 
                if 'averaged' not in self.d1s[sn].keys():
                    print("1d data not available.")
                    continue
                self.d1s[sn]['averaged'].save("%s%s_%c.dat"%(path,sn,'a'), debug=debug, 
                                              footer=self.md_string(sn))
                
    def load_data_mp(self, *args, **kwargs):
        print('load_data_mp() will be deprecated. use load_data() instead.')
        self.load_data(*args, **kwargs)
            
    def load_data(self, update_only=False, detectors=None,
           reft=-1, save_1d=False, save_merged=False, debug=False, N=8, max_c_size=0):
        """ assume multiple samples, parallel-process by sample
            use Pool to limit the number of processes; 
            access h5 group directly in the worker process
        """
        if debug is True:
            print("start processing: load_data()")
            t1 = time.time()
        
        fh5 = self.fh5
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        
        results = {}
        pool = mp.Pool(N)
        jobs = []
        
        for sn in self.samples:
            if sn not in list(self.attrs.keys()):
                self.attrs[sn] = {}
            if 'buffer' in list(fh5[sn].attrs):
                self.buffer_list[sn] = fh5[sn].attrs['buffer'].split('  ')
            if update_only and sn in list(self.d1s.keys()):
                self.load_d1s(sn)   # load processed data saved in the file
                continue
                                    
            self.d1s[sn] = {}
            results[sn] = {}
            dset = fh5["%s/primary/data" % sn]
            
            s = dset["%s" % self.det_name[self.detectors[0].extension]].shape
            if len(s)==3 or len(s)==4:
                n_total_frames = s[0]
            else:
                raise Exception("don't know how to handle shape:", )
            if n_total_frames<N*N/2:
                Np = 1
                c_size = N
            else:
                Np = N
                c_size = int(n_total_frames/N)
                if max_c_size>0 and c_size>max_c_size:
                    Np = int(n_total_frames/max_c_size)+1
                    c_size = int(n_total_frames/Np)
                    
            # process data in group in hope to limit memory use
            # the raw data could be stored in a 1d or 2d array
            if detectors is None:
                detectors = self.detectors
            for i in range(Np):
                if i==Np-1:
                    nframes = n_total_frames - c_size*(Np-1)
                else:
                    nframes = c_size
                    
                if len(s)==3:
                    images = {}
                    for det in detectors:
                        gn = f'{self.det_name[det.extension]}'
                        images[det.extension] = dset[gn][i*c_size:i*c_size+nframes]    

                    if N>1: # multi-processing, need to keep track of total number of active processes                    
                        job = pool.map_async(proc_d1merge, [(images, sn, nframes, i*c_size, debug,
                                                             detectors, self.qgrid, reft, save_1d, save_merged)])
                        jobs.append(job)
                    else: # serial processing
                        [sn, fr1, data] = proc_d1merge((images, sn, nframes, i*c_size, debug, 
                                                        detectors, self.qgrid, reft, save_1d, save_merged)) 
                        results[sn][fr1] = data                
                else: # len(s)==4
                    for j in range(s[1]):
                        images = {}
                        for det in detectors:
                            gn = f'{self.det_name[det.extension]}'
                            images[det.extension] = dset[gn][i*c_size:i*c_size+nframes, j]
                        if N>1: # multi-processing, need to keep track of total number of active processes
                            job = pool.map_async(proc_d1merge, [(images, sn, nframes, i*c_size+j*s[0], debug,
                                                                 detectors, self.qgrid, reft, save_1d, save_merged)])
                            jobs.append(job)
                        else: # serial processing
                            [sn, fr1, data] = proc_d1merge((images, sn, nframes, i*c_size+j*s[0], debug, 
                                                            detectors, self.qgrid, reft, save_1d, save_merged)) 
                            results[sn][fr1] = data                

        if N>1:             
            for job in jobs:
                [sn, fr1, data] = job.get()[0]
                results[sn][fr1] = data
                print("data received: sn=%s, fr1=%d" % (sn,fr1) )
            pool.close()
            pool.join()

        for sn in self.samples:
            if sn not in results.keys():
                continue
            data = {}
            frns = list(results[sn].keys())
            frns.sort()
            for k in results[sn][frns[0]].keys():
                data[k] = []
                for frn in frns:
                    data[k].extend(results[sn][frn][k])
            self.d1s[sn] = data
        
        self.save_d1s(debug=debug)
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))

    def load_data0(self, update_only=False, 
                   reft=-1, save_1d=False, save_merged=False, debug=False, N=8):
        """ assume multiple samples, parallel-process by sample
            if update_only is true, only create 1d data for new frames (empty self.d1s)
        """
        if debug is True:
            print("start processing: load_data()")
            t1 = time.time()
        
        fh5 = self.fh5
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        
        processes = []
        queue_list = []
        results = {}
        for sn in self.samples:
            if sn not in list(self.attrs.keys()):
                self.attrs[sn] = {}
            if 'buffer' in list(fh5[sn].attrs):
                self.buffer_list[sn] = fh5[sn].attrs['buffer'].split('  ')
            if update_only and sn in list(self.d1s.keys()):
                self.load_d1s(sn)   # load processed data saved in the file
                continue
                                    
            self.d1s[sn] = {}
            results[sn] = {}
            images = {}
            for det in self.detectors:
                ti = fh5["%s/primary/data/%s" % (sn, self.det_name[det.extension])][...]  # ].value
                if len(ti.shape)>3:
                    ti = ti.reshape(ti.shape[-3:])      # quirk of suitcase
                images[det.extension] = ti
            
            n_total_frames = len(images[self.detectors[0].extension])
            if N>1: # multi-processing
                if n_total_frames<N*N/2:
                    Np = 1
                    c_size = N
                else:
                    Np = N
                    c_size = int(n_total_frames/N)
                for i in range(Np):
                    if i==Np-1:
                        nframes = n_total_frames - c_size*(Np-1)
                    else:
                        nframes = c_size
                    que = mp.Queue()
                    queue_list.append(que)
                    th = mp.Process(target=proc_sample, 
                                    args=(que, images, sn, nframes,
                                          self.detectors, self.qgrid, reft, save_1d, save_merged, debug, i*c_size) )
                    th.start()
                    processes.append(th)
            else: # serial processing
                [sn, fr1, data] = proc_sample(None, images, sn, n_total_frames,
                                              self.detectors, self.qgrid, reft, save_1d, save_merged, debug) 
                self.d1s[sn] = data                

        if N>1:             
            for que in queue_list:
                [sn, fr1, data] = que.get()
                results[sn][fr1] = data
                print("data received: sn=%s, fr1=%d" % (sn,fr1) )
            for th in processes:
                th.join()

            for sn in self.samples:
                if sn not in results.keys():
                    continue
                data = {}
                frns = list(results[sn].keys())
                frns.sort()
                for k in results[sn][frns[0]].keys():
                    data[k] = []
                    for frn in frns:
                        data[k].extend(results[sn][frn][k])
                self.d1s[sn] = data
        
        self.save_d1s(debug=debug)
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            

class h5sol_HPLC(h5xs):
    """ single sample (not required, but may behave unexpectedly when there are multiple samples), 
        many frames; frames can be added gradually (not tested)
    """ 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbuf = None
        self.updating = False   # this is set to True when add_data() is active
        
    def add_data(self, ):
        """ watch the given path
            update self.d1s when new files are found
            save to HDF when the scan is done
        """
        pass
    
    def process_sample_name(self, sn, debug=False):
        #fh5 = h5py.File(self.fn, "r+")
        fh5 = self.fh5
        self.samples = lsh5(fh5, top_only=True, silent=(debug is not True))
        if sn==None:
            sn = self.samples[0]
        elif sn not in self.samples:
            raise Exception(sn, "not in the sample list.")
        
        return fh5,sn 
        
    def load_d1s(self):
        super().load_d1s(self.samples[0])
        # might need to update meta data??
        
    def normalize_int(self, ref_trans=-1):
        """ 
        """
        sn = self.samples[0]        
        if 'merged' not in self.d1s[sn].keys():
            raise Exception(f"{sn}: merged data must exist before normalizing intensity.")

        max_trans = np.max([d1.trans for d1 in self.d1s[sn]['merged']])
        if max_trans<=0:
            raise Exception(f"{sn}: run set_trans() first, or the beam may be off during data collection.")
        if ref_trans<0:
            ref_trans=max_trans
            
        for d1 in self.d1s[sn]['merged']:
            d1.scale(ref_trans/d1.trans)
        
    def process(self, update_only=False, ext_trans=False,
                reft=-1, save_1d=False, save_merged=False, 
                filter_data=False, debug=False, N=8, max_c_size=0):
        """ load data from 2D images, merge, then set transmitted beam intensity
        """

        self.load_data(update_only=update_only, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N, max_c_size=max_c_size)
        
        # This should account for any beam intensity fluctuation during the HPLC run. While
        # typically for solution scattering the water peak intensity is relied upon for normalization,
        # it could be problematic if sample scattering has features at high q.  
        if ext_trans and self.transField is not None:
            self.set_trans(transMode=trans_mode.external)
        else:
            self.set_trans(transMode=trans_mode.from_waxs) 
        self.normalize_int()

    def subtract_buffer_SVD(self, excluded_frames_list, sn=None, sc_factor=0.995,
                            gaussian_filter_width=None,
                            Nc=5, poly_order=8, smoothing_factor=0.04, fit_with_polynomial=False,
                            plot_fit=False, ax1=None, ax2=None, debug=False):
        """ perform SVD background subtraction, use Nc eigenvalues 
            poly_order: order of polynomial fit to each eigenvalue
            gaussian_filter width: sigma value for the filter, e.g. 1, or (0.5, 3)
        """
        fh5,sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        if isinstance(poly_order, int):
            poly_order = poly_order*np.ones(Nc, dtype=np.int)
        elif isinstance(poly_order, list):
            if len(poly_order)!=Nc:
                raise Exception(f"the length of poly_order ({poly_order}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid poly_order: {poly_order}")

        if isinstance(smoothing_factor, float) or isinstance(smoothing_factor, int):
            smoothing_factor = smoothing_factor*np.ones(Nc, dtype=np.float)
        elif isinstance(poly_order, list):
            if len(smoothing_factor)!=Nc:
                raise Exception(f"the length of smoothing_factor ({smoothing_factor}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid smoothing_factor: {smoothing_factor}")
                    
        nf = len(self.d1s[sn]['merged'])
        all_frns = list(range(nf))
        ex_frns = []
        for r in excluded_frames_list.split(','):
            if r=="":
                break
            r1,r2 = np.fromstring(r, dtype=int, sep='-')
            ex_frns += list(range(r1,r2))
        bkg_frns = list(set(all_frns)-set(ex_frns))
        
        dd2s = np.vstack([d1.data for d1 in self.d1s[sn]['merged']]).T
        if gaussian_filter_width is not None:
            dd2s = gaussian_filter(dd2s, sigma=gaussian_filter_width)
        dd2b = np.vstack([dd2s[:,i] for i in bkg_frns]).T
        
        U, s, Vh = svd(dd2b.T, full_matrices=False)
        s[Nc:] = 0

        Uf = []
        # the time-dependence of the eigen values are fitted to fill the gap (excluded frames) 
        # polynomial fits will likely produce unrealistic fluctuations
        # cubic (default, k=3) spline fits with smoothing factor provides better control
        #     smoothing factor: # of knots are added to reduce fitting error below the s factor??
        for i in range(Nc):
            if fit_with_polynomial:
                Uf.append(np.poly1d(np.polyfit(bkg_frns, U[:,i], poly_order[i])))
            else:
                Uf.append(UnivariateSpline(bkg_frns, U[:,i], s=smoothing_factor[i]))
        Ub = np.vstack([f(all_frns) for f in Uf]).T
        dd2c = np.dot(np.dot(Ub, np.diag(s[:Nc])), Vh[:Nc,:]).T

        if plot_fit:
            if ax1 is None:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
            for i in reversed(range(Nc)):
                ax1.plot(bkg_frns, np.sqrt(s[i])*U[:,i], '.') #s[i]*U[:,i])
            for i in reversed(range(Nc)):
                ax1.plot(all_frns, np.sqrt(s[i])*Uf[i](all_frns)) #s[i]*U[:,i])  
            ax1.set_xlim(0, nf)
            ax1.set_xlabel("frame #")
            if ax2 is not None:
                for i in reversed(range(Nc)):
                    ax2.plot(self.qgrid, np.sqrt(s[i])*Vh[i]) #s[i]*U[:,i])
                ax2.set_xlabel("q")                

        self.attrs[sn]['sc_factor'] = sc_factor
        self.attrs[sn]['svd excluded frames'] = excluded_frames_list
        self.attrs[sn]['svd parameter Nc'] = Nc
        self.attrs[sn]['svd parameter poly_order'] = poly_order
        if 'subtracted' in self.d1s[sn].keys():
            del self.d1s[sn]['subtracted']
        self.d1s[sn]['subtracted'] = []
        dd2s -= dd2c*sc_factor
        for i in range(nf):
            d1c = copy.deepcopy(self.d1s[sn]['merged'][i])
            d1c.data = dd2s[:,i]
            self.d1s[sn]['subtracted'].append(d1c)
            
        self.save_d1s(sn, debug=debug)

        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))

    def subtract_buffer(self, buffer_frame_range, sample_frame_range=None, first_frame=0, 
                        sn=None, update_only=False, 
                        sc_factor=1., show_eb=False, debug=False):
        """ buffer_frame_range should be a list of frame numbers, could be range(frame_s, frame_e)
            if sample_frame_range is None: subtract all dataset; otherwise subtract and test-plot
            update_only is not used currently
            first_frame:    duplicate data in the first few frames subtracted data from first_frame
                            this is useful when the beam is not on for the first few frames
        """

        fh5,sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        if type(buffer_frame_range) is str:
            f1,f2 = buffer_frame_range.split('-')
            buffer_frame_range = range(int(f1), int(f2))
            
        listb  = [self.d1s[sn]['merged'][i] for i in buffer_frame_range]
        listbfn = buffer_frame_range
        if len(listb)>1:
            d1b = listb[0].avg(listb[1:], debug=debug)
        else:
            d1b = copy.deepcopy(listb[0])           
            
        if sample_frame_range==None:
            # perform subtraction on all data and save listbfn, d1b
            self.attrs[sn]['buffer frames'] = listbfn
            self.attrs[sn]['sc_factor'] = sc_factor
            self.d1s[sn]['buf average'] = d1b
            if 'subtracted' in self.d1s[sn].keys():
                del self.d1s[sn]['subtracted']
            self.d1s[sn]['subtracted'] = []
            for d1 in self.d1s[sn]['merged']:
                d1t = d1.bkg_cor(d1b, plot_data=False, debug=debug, sc_factor=sc_factor)
                self.d1s[sn]['subtracted'].append(d1t) 
            if first_frame>0:
                for i in range(first_frame):
                    self.d1s[sn]['subtracted'][i].data = self.d1s[sn]['subtracted'][first_frame].data
            self.save_d1s(sn, debug=debug)   # save only subtracted data???
        else:
            lists  = [self.d1s[sn]['merged'][i] for i in sample_frame_range]
            if len(listb)>1:
                d1s = lists[0].avg(lists[1:], debug=debug)
            else:
                d1s = copy.deepcopy(lists[0])
            sample_sub = d1s.bkg_cor(d1b, plot_data=True, debug=debug, sc_factor=sc_factor, show_eb=show_eb)
            return sample_sub
        
        #if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
        #if sn not in list(self.buffer_list.keys()): continue

        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    def get_chromatogram(self, sn, q_ranges=[[0.02,0.05]], flowrate=0, plot_merged=False,
                 calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True):
        """ returns data to be plotted in the chromatogram
        """
        if 'subtracted' in self.d1s[sn].keys() and plot_merged==False:
            dkey = 'subtracted'
        elif 'merged' in self.d1s[sn].keys():
            if plot_merged==False:
                print("subtracted data not available. plotting merged data instead ...")
            dkey = 'merged'
        else:
            raise Exception("processed data not present.")
            
        data = self.d1s[sn][dkey]
        nd = len(data)
        #qgrid = data[0].qgrid
        
        ts = self.fh5[sn+'/primary/time'][...]  # self.fh5[sn+'/primary/time'].value
        if len(ts)==1:
            # there is only one time stamp for multi-frame data collection
            cfg = json.loads(self.fh5[f'{sn}/primary'].attrs['configuration'])
            k = list(cfg.keys())[0]
            ts = np.arange(len(data)) * cfg[k]['data'][f"{k}_cam_acquire_period"]
            
        idx = [(self.qgrid>i_minq) & (self.qgrid<i_maxq) for [i_minq,i_maxq] in q_ranges]
        nq = len(idx)
        
        d_t = np.zeros(nd)
        d_i = np.zeros((nq, nd))
        d_rg = np.zeros(nd)
        d_s = []
        
        for i in range(len(data)):
            ti = (ts[i]-ts[0])/60
            #if flowrate>0:
            #    ti*=flowrate
            d_t[i] = ti
                
            for j in range(nq): 
                d_i[j][i] = data[i].data[idx[j]].sum()
            ii = np.max([d_i[j][i] for j in range(nq)])
            d_s.append(data[i].data)

            if ii>thresh and calc_Rg and dkey=='subtracted':
                i0,rg,_ = data[i].plot_Guinier(qs, qe, fix_qe=fix_qe, no_plot=True)
                d_rg[i] = rg
    
        # read HPLC data directly from HDF5
        hplc_grp = self.fh5[sn+"/hplc/data"]
        fields = lsh5(self.fh5[sn+'/hplc/data'], top_only=True, silent=True)
        d_hplc = {}
        for fd in fields:
            d_hplc[fd] = self.fh5[sn+'/hplc/data/'+fd][...].T   # self.fh5[sn+'/hplc/data/'+fd].value.T
    
        return dkey,d_t,d_i,d_hplc,d_rg,np.vstack(d_s).T
    
            
    def plot_data(self, sn=None, 
                  q_ranges=[[0.02,0.05]], logROI=False, markers=['bo', 'mo', 'co', 'yo'],
                  flowrate=-1, plot_merged=False,
                  ymin=-1, ymax=-1, offset=0, uv_scale=1, showFWHM=False, 
                  calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True,
                  plot2d=True, logScale=True, clim=[1.e-3, 10.],
                  show_hplc_data=[True, False],
                  export_txt=False, debug=False, 
                  fig_w=8, fig_h1=2, fig_h2=3.5, ax1=None, ax2=None):
        """ plot "merged" if no "subtracted" present
            q_ranges: a list of [q_min, q_max], within which integrated intensity is calculated 
            export_txt: export the scattering-intensity-based chromatogram
            
        """
        
        if ax1 is None:
            if plot2d:
                fig = plt.figure(figsize=(fig_w, fig_h1+fig_h2))
                hfrac = 0.82                
                ht2 = fig_h1/(fig_h1+fig_h2)
                box1 = [0.1, ht2+0.05, hfrac, (0.95-ht2)*hfrac] # left, bottom, width, height
                box2 = [0.1, 0.02, hfrac, ht2*hfrac]
                ax1 = fig.add_axes(box1)
            else:
                plt.figure(figsize=(fig_w, fig_h2))
                ax1 = plt.gca()
        ax1a = ax1.twiny()
        ax1b = ax1.twinx()
        
        fh5,sn = self.process_sample_name(sn, debug=debug)
        if flowrate<0:  # get it from metadata
            md = self.md_dict(sn, md_keys=['HPLC'])
            if "HPLC" in md.keys():
                flowrate = float(md["HPLC"]["Flow Rate (ml_min)"])
            else: 
                flowrate = 0.5
        dkey,d_t,d_i,d_hplc,d_rg,d_s = self.get_chromatogram(sn, q_ranges=q_ranges, 
                                                             flowrate=flowrate, plot_merged=plot_merged, 
                                                             calc_Rg=calc_Rg, thresh=thresh, 
                                                             qs=qs, qe=qe, fix_qe=fix_qe)
        data = self.d1s[sn][dkey]
        nq = len(q_ranges)
        
        if ymin == -1:
            ymin = np.min([np.min(d_i[j]) for j in range(nq)])
        if ymax ==-1:
            ymax = np.max([np.max(d_i[j]) for j in range(nq)])
        if logROI:
            pl_ymax = 1.5*ymax
            pl_ymin = 0.8*np.max([ymin, 1e-2])
        else:
            pl_ymax = ymax+0.05*(ymax-ymin)
            pl_ymin = ymin-0.05*(ymax-ymin)

        if export_txt:
            # export the scattering-intensity-based chromatogram
            for j in range(nq):
                np.savetxt(f'{sn}.chrome_{j}', np.vstack((d_t, d_i[j])).T, "%12.3f")
            
        for j in range(nq):
            ax1.plot(d_i[j], 'w-')
        ax1.set_xlabel("frame #")
        ax1.set_xlim((0,len(d_i[0])))
        ax1.set_ylim(pl_ymin, pl_ymax)
        ax1.set_ylabel("intensity")
        if logROI:
            ax1.set_yscale('log')

        i = 0 
        for k,dc in d_hplc.items():
            if show_hplc_data[i]:
                ax1a.plot(np.asarray(dc[0])+offset,
                         ymin+dc[1]/np.max(dc[1])*(ymax-ymin)*uv_scale, label=k)
            i += 1
            #ax1a.set_ylim(0, np.max(dc[0][2]))

        if flowrate>0:
            ax1a.set_xlabel("volume (mL)")
        else:
            ax1a.set_xlabel("time (minutes)")
        for j in range(nq):
            ax1a.plot(d_t, d_i[j], markers[j], markersize=5, label=f'x-ray ROI #{j+1}')
        ax1a.set_xlim((d_t[0],d_t[-1]))
        leg = ax1a.legend(loc='upper left', fontsize=9, frameon=False)

        if showFWHM and nq==1:
            half_max=(np.amax(d_i[0])-np.amin(d_i[0]))/2 + np.amin(d_i[0])
            s = splrep(d_t, d_i[0] - half_max)
            roots = sproot(s)
            fwhm = abs(roots[1]-roots[0])
            print(roots[1],roots[0],half_max)
            if flowrate>0:
                print("X-ray cell FWHMH =", fwhm, "ml")
            else:
                print("X-ray cell FWHMH =", fwhm, "min")
            ax1a.plot([roots[0], roots[1]],[half_max, half_max],"k-|")

        if calc_Rg and dkey=='subtracted':
            d_rg = np.asarray(d_rg)
            max_rg = np.max(d_rg)
            d_rg[d_rg==0] = np.nan
            ax1b.plot(d_rg, 'r.', label='rg')
            ax1b.set_xlim((0,len(d_rg)))
            ax1b.set_ylim((0, max_rg*1.05))
            ax1b.set_ylabel("Rg")
            leg = ax1b.legend(loc='center left', fontsize=9, frameon=False)
        else:
            ax1b.yaxis.set_major_formatter(plt.NullFormatter())

        if plot2d:
            if ax2 is None:
                ax2 = fig.add_axes(box2)
            ax2.tick_params(axis='x', top=True)
            ax2.xaxis.set_major_formatter(plt.NullFormatter())

            d2 = d_s + clim[0]/2
            ext = [0, len(data), len(self.qgrid), 0]
            asp = len(d_t)/len(self.qgrid)/(fig_w/fig_h1)
            if logScale:
                im = ax2.imshow(np.log(d2), extent=ext, aspect="auto") 
                im.set_clim(np.log(clim))
            else:
                im = ax2.imshow(d2, extent=ext, aspect="auto") 
                im.set_clim(clim)
            
            gpindex,gpvalues,gplabels = qgrid_labels(self.qgrid)
            ax2.set_yticks(gpindex)
            ax2.set_yticklabels(gplabels)
            ax2.set_ylabel('q')            
            
            ax2a = ax2.twinx()
            ax2a.set_ylim(len(self.qgrid)-1, 0)
            ax2a.set_ylabel('point #')
            

        #plt.tight_layout()
        #plt.show()
        
    def bin_subtracted_frames(self, sn=None, frame_range=None, first_frame=0, last_frame=-1, weighted=True,
                              plot_data=True, fig=None, qmax=0.5, qs=0.01,
                              save_data=False, path="", debug=False): 
        """ this is typically used after running subtract_buffer_SVD()
            the frames are specified by either first_frame and last_frame, or frame_range, e.g. "50-60"
            if path is used, be sure that it ends with '/'
        """
        fh5,sn = self.process_sample_name(sn, debug=debug)
        
        if plot_data:
            if fig is None:
                fig = plt.figure()            
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        for fr in frame_range.strip(' ').split(','):    
            if fr is None:
                continue
            f1,f2 = fr.split('-')
            first_frame = int(f1)
            last_frame = int(f2)
            if last_frame<first_frame:
                last_frame=len(self.d1s[sn]['subtracted'])
            if debug is True:
                print(f"binning frames {fr}: first_frame={first_frame}, last_frame={last_frame}")
            d1s0 = copy.deepcopy(self.d1s[sn]['subtracted'][first_frame])
            if last_frame>first_frame+1:
                d1s0 = d1s0.avg(self.d1s[sn]['subtracted'][first_frame+1:last_frame], 
                                weighted=weighted, debug=debug)
            if save_data:
                d1s0.save(f"{path}{sn}_{first_frame:04d}-{last_frame-1:04d}s.dat", debug=debug)
            if plot_data:
                ax1.semilogy(d1s0.qgrid, d1s0.data)
                ax1.errorbar(d1s0.qgrid, d1s0.data, d1s0.err)
                ax1.set_xlim(0,qmax)
                i0,rg,_ = d1s0.plot_Guinier(qs=qs, ax=ax2)
            #print(f"I0={i0:.2g}, Rg={rg:.2f}")

        if plot_data:
            plt.tight_layout()   
            
        return d1s0
    
        
    def export_txt(self, sn=None, first_frame=0, last_frame=-1, save_subtracted=True,
                   averaging=False, plot_averaged=False, ax=None, path="",
                   debug=False):
        """ if path is used, be sure that it ends with '/'
        """
        fh5,sn = self.process_sample_name(sn, debug=debug)
        if save_subtracted:
            if 'subtracted' not in self.d1s[sn].keys():
                print("subtracted data not available.")
                return
            dkey = 'subtracted'
        else:
            if 'merged' not in self.d1s[sn].keys():
                print("1d data not available.")
                return
            dkey = 'merged'
        if last_frame<first_frame:
            last_frame=len(self.d1s[sn][dkey])

        d1s = self.d1s[sn][dkey][first_frame:last_frame]
        if averaging:
            d1s0 = copy.deepcopy(d1s[0])
            if len(d1s)>1:
                d1s0.avg(d1s[1:], weighted=True, plot_data=plot_averaged, ax=ax, debug=debug)
            d1s0.save(f"{path}{sn}_{first_frame:04d}-{last_frame-1:04d}{dkey[0]}.dat", 
                      debug=debug, footer=self.md_string(sn, md_keys=['HPLC']))
        else:
            for i in range(len(d1s)):
                d1s[i].save(f"{path}{sn}_{i+first_frame:04d}{dkey[0]}.dat", 
                            debug=debug, footer=self.md_string(sn, md_keys=['HPLC']))                    

        
class h5sol_HT(h5xs):
    """ multiple samples, not many frames per sample
    """    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d1b = {}   # buffer data used, could be averaged from multiple samples
        self.buffer_list = {}    
        
    def add_sample(self, db, uid):
        """ add another group to the HDF5 file
            only works at the beamline
        """
        header = db[uid]
        
    def load_d1s(self, sn=None):
        if sn==None:
            samples = self.samples
        elif isinstance(sn, str):
            samples = [sn]
        else:
            samples = sn
        
        for sn in samples:
            super().load_d1s(sn)
            if 'buffer' in list(self.fh5[sn].attrs.keys()):
                self.buffer_list[sn] = self.fh5[sn].attrs['buffer'].split()
        
    def assign_buffer(self, buf_list, debug=False):
        """ buf_list should be a dict:
            {"sample_name": "buffer_name",
             "sample_name": ["buffer1_name", "buffer2_name"],
             ...
            }
            anything missing is considered buffer
        """
        for sn in list(buf_list.keys()):
            if isinstance(buf_list[sn], str):
                self.buffer_list[sn] = [buf_list[sn]]
            else:
                self.buffer_list[sn] = buf_list[sn]
            #self.attrs[sn]['buffer'] = self.buffer_list[sn] 
        
        if debug is True:
            print('updating buffer assignments')
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                self.fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
        self.fh5.flush()               

    def change_buffer(self, sample_name, buffer_name):
        """ buffer_name could be just a string (name) or a list of names 
        """
        if sample_name not in self.samples:
            raise Exception(f"invalid sample name: {sample_name}")
        if isinstance(buffer_name, str):
            buffer_name = [buffer_name]
        for b in buffer_name:
            if b not in self.samples:
                raise Exception(f"invalid buffer name: {b}")
        
        self.buffer_list[sample_name] = buffer_name 
        self.fh5[sample_name].attrs['buffer'] = '  '.join(buffer_name)
        self.fh5.flush()               
        self.subtract_buffer(sample_name)
        
    def update_h5(self, debug=False):
        """ raw data are updated using add_sample()
            save sample-buffer assignment
            save processed data
        """
        #fh5 = h5py.File(self.fn, "r+")
        if debug is True:
            print("updating 1d data and buffer info ...") 
        fh5 = self.fh5
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
            self.save_d1s(sn, debug=debug)
        #fh5.flush()                           
        
    def process(self, detectors=None, update_only=False,
                reft=-1, sc_factor=1., save_1d=False, save_merged=False, 
                filter_data=True, debug=False, N = 1):
        """ does everything: load data from 2D images, merge, then subtract buffer scattering
        """
        self.load_data(update_only=update_only, detectors=detectors, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N)
        self.set_trans(transMode=trans_mode.from_waxs)
        self.average_samples(update_only=update_only, filter_data=filter_data, debug=debug)
        self.subtract_buffer(update_only=update_only, sc_factor=sc_factor, debug=debug)
        
    def average_samples(self, **kwargs):
        """ if update_only is true: only work on samples that do not have "merged' data
            selection: if None, retrieve from dataset attribute
        """
        super().average_d1s(**kwargs)
            
    def subtract_buffer(self, samples=None, update_only=False, sc_factor=1., debug=False):
        """ if update_only is true: only work on samples that do not have "subtracted' data
            sc_factor: if <0, read from the dataset attribute
        """
        if samples is None:
            samples = list(self.buffer_list.keys())
        elif isinstance(samples, str):
            samples = [samples]

        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()
        for sn in samples:
            if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
            if sn not in list(self.buffer_list.keys()): continue
            
            bns = self.buffer_list[sn]
            if isinstance(bns, str):
                self.d1b[sn] = self.d1s[bns]['averaged']  # ideally this should be a link
            else:
                self.d1b[sn] = self.d1s[bns[0]]['averaged'].avg([self.d1s[bn]['averaged'] for bn in bns[1:]], 
                                                                debug=debug)
            if sc_factor is "auto":
                sf = estimate_scaling_factor(self.d1s[sn]['averaged'], self.d1b[sn])
                # Data1d.bkg_cor() normalizes trans first before applying sc_factor
                # in contrast the estimated 
                sf /= self.d1s[sn]['averaged'].trans/self.d1b[sn].trans
                if debug is not "quiet":
                    print(f"setting sc_factor for {sn} to {sf:.4f}")
                self.attrs[sn]['sc_factor'] = sf
            elif sc_factor>0:
                self.attrs[sn]['sc_factor'] = sc_factor
                sf = sc_factor
            else:
                sf = self.attrs[sn]['sc_factor']
            self.d1s[sn]['subtracted'] = self.d1s[sn]['averaged'].bkg_cor(self.d1b[sn], 
                                                                          sc_factor=sf, debug=debug)

        self.update_h5()  #self.fh5.flush()
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
                
    def plot_sample(self, *args, **kwargs):
        """ show_subtracted:
                work only if sample is background-subtracted, show the subtracted result
            show_subtraction: 
                if True, show sample and boffer when show_subtracted
            show_overlap: 
                also show data in the overlapping range from individual detectors
                only allow if show_subtracted is False
        """
        super().plot_d1s(*args, **kwargs)
     
    def export_txt(self, *args, **kwargs):
        super().export_d1s(*args, **kwargs)
