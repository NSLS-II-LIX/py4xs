# need to have a more uniform method to exchange (pack/unpack) 1D and 2D PROCESSED data with hdf5
# type of data: Data1d, MatrixWithCoordinates (not just simple numpy arrays)
import pylab as plt
import h5py
import numpy as np
import time,datetime
import os,copy,subprocess,re
import json,pickle,fabio
import multiprocessing as mp
import numbers

from py4xs.slnxs import Data1d,average,filter_by_similarity,trans_mode,estimate_scaling_factor
from py4xs.utils import common_name,max_len,Schilling_p_value
from py4xs.detector_config import create_det_from_attrs
from py4xs.local import det_names,det_model,beamline_name,incident_monitor,transmitted_monitor
from py4xs.data2d import Data2d,MatrixWithCoords,DataType
from py4xs.plot import show_data,show_data_qxy,show_data_qphi
from py4xs.utils import run
from itertools import combinations

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import UnivariateSpline as uspline
from scipy.integrate import simpson

from functools import wraps

USE_DASK = True
try:
    from dask.distributed import Client,LocalCluster
    import dask.array as da
except:
    USE_DASK = False

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

def get_det_timestamps(grp, trigger, exp, err_band=5., debug=False):
    """ in fly scans, the timestamps comes from the XPS controller. sometimes the timestamps appear to be off
        the timestamp on the detector image is used as a fallback
        
        also check the length of the timestamp data, the total number of data points should agree of 
        number of frames in the scattering data
    """
    ts0 = grp[f'timestamps/{trigger}'][...].flatten()
    dns = [dn for dn in grp['timestamps'].keys() if 'image' in dn]
    if len(dns)<1:
        raise Exception("cannot find detector data in {grp}")
    tsd = grp[f'timestamps/{dns[0]}'][...]

    dshape = grp[f'data/{dns[0]}'].shape[:-2]
    if len(dshape)>1:
        if len(dshape)>2:
            raise Exception(f"Don't know how to handle data shape {dshape}")
        nimg = dshape[0]*dshape[1]
        ts0 = ts0.flatten()
    else:
        nimg = dshape[0]
    if len(ts0) != nimg:
        print(f"Warning: mistached timestamp length: {len(ts0)} vs {nimg}")
        #raise Exception(f"mistached timestamp length: {len(ts0)} vs {dshape}")
    
    if debug:
        print(f"time stamp from {trigger} ends at {ts0[-1]} .")
        print(f"time stamp from {dns[0]} ends at {tsd[-1]} .")
        print(f"off by {ts0[-1]-tsd[-1]:.3f}")
        print(f"size of the timestamp data: {len(ts0)}")
        print(f"size of the scattering data : {nimg}")
    
    if len(ts0)<nimg:
        print("Warning: more scattering patterns than recorded triggers, repeating the last triggers ...")
        ts01 = ts0[:nimg-len(ts0)]
        ts0 = np.append(ts0, ts01-ts01[0]+ts0[-1]+exp)
        #raise Exception("more scattering patterns than recorded triggers ...")
    elif len(ts0)>nimg:
        print("Warning: more triggers than existing scattering patterns, ignore the extra triggers ... ")
        ts0 = ts0[:nimg]
    
    dt = ts0[-1]-tsd[-1]
    if np.fabs(dt)>err_band:
        print(f"Warning: strange trigger time stamps (different by {dt:.2f} s), falling back to {dns[0]} data")
        ts0 += tsd[-1]-exp-ts0[-1]

    return ts0    
    
def synch_mon(ts0, em0, ts, em, dt="auto", Ns=8, plot=False):
    """ revise the timestamp ts (np array) to match the monitor em to em0
        look for the most significant change, i.e. when the shutter opens 
        
        if dt is a numberical value, simply add that value to ts
        if dt is "auto", in principle both em's should see the shutter openning when the scan starts
        
        Ns specifies how much smoothing will be applied to the em data before synching
    """
    if isinstance(dt, numbers.Number):
        ts += ts0[0]-ts[0]+dt
        return dt
    if dt!="auto": 
        return 0   # don't know what to do
    
    ts += ts0[0]-ts[0]
    df = np.diff(em)
    df0 = np.diff(em0)
    df /= np.max(df)
    df0 /= np.max(df0)

    tt = np.linspace(-Ns, Ns, 2*Ns+1)
    smth = np.exp(-tt**2/Ns)
    df = np.convolve(df, smth, "same")
    df0 = np.convolve(df0, smth, "same")
    idx = df>0.4*np.max(df)
    idx0 = df0>0.4*np.max(df0)
    ats = np.average(ts[1:][idx], weights=df[idx])
    ats0 = np.average(ts0[1:][idx0], weights=df0[idx0])
    
    if plot:
        plt.figure()
        plt.plot(ts[1:][idx], df[idx])
        plt.plot(ts0[1:][idx0], df0[idx0])
        plt.bar([ats], [2], width=0.01)
        plt.bar([ats0], [2], width=0.01)

    dt = ats0-ats
    ts += dt
    return dt  
    
def integrate_mon(em, ts, ts0, exp, extend_mon_stream, ts_pos_in_frame="start", debug=False):
    """ integrate monitor counts
        monitor counts are given by em with timestamps ts
        ts0 is the timestamps on the exposures, with duration of exp
        if extend_mon_stream, repeat the ts data at the beginning and the end to enclose the range of ts0 
        assume ts and ts0 are 1d arrays
        
        ts_pos_in_frame: "start" or "end"
            whether the timestamp correspond to the start of the exposure or the end of the exposure
    """
    if debug:
        print(f"integrating monitor counts, time stamp offset mon-tr={ts[0]-ts0[0]:.1f}")
    if extend_mon_stream: 
        em = np.concatenate(([em[0]], em, [em[-1]]))
        ts = np.concatenate(([np.min([ts0[0], ts[0]])-exp*1.5], ts, [np.max([ts0[-1], ts[-1]])+exp*1.5]))
    ffe = interp1d(ts, em) 
    em0 = []
    
    ti0 = {"start": 0, "end": -exp}
    ti1 = {"start": exp, "end": 0}
    
    for t in ts0:
        tt = np.concatenate(([t+ti0[ts_pos_in_frame]], 
                              ts[(ts>t+ti0[ts_pos_in_frame]) & (ts<t+ti1[ts_pos_in_frame])], 
                              [t+ti1[ts_pos_in_frame]])) 
        ee = ffe(tt)
        em0.append(simpson(ee, tt))
    return np.asarray(em0)/exp

def get_monitor_counts(grp, monitorName):
    """ look under a data group (grp) that belong to a specific sample, find the stream that contains fieldName
        caluclate the monitor counts based on the given timestamps (ts) and exposure time
    """
    strn = None
    for stream in list(grp):
        if not 'data' in list(grp[stream]):
            continue
        for fieldName in list(grp[stream]["data"]):
            if monitorName in fieldName:
                strn = stream
                data = grp[strn]["data"][fieldName][...]
                #ts = grp[strn]["timestamps"][fieldName][...]   # this filed seems to have occasional issues 
                ts = grp[strn]["time"][...]   # this is consistent with data from databroker
                break
    if strn is None:
        raise Exception(f"could not find the stream that contains {monitorName}.")

    # in the case of timeseries data, the time stamps need to be lengthened
    n = int(len(data)/len(ts))
    if n>1:
        # sometimes the first two timestamps are the same; based on the data, this seems to be 
        # an error, and the first timestamp should to be forward adjusted 
        if ts[0]==ts[1]:
            ts[0] -= (ts[2]-ts[1])
            #data = np.reshape(data, [-1, len(ts)])
            #ts = ts[1:]
            #data = data[1:, :]
            #data = data.flatten()
        dt = np.average(np.diff(ts))  # the intervals appear to be constant in the data
        # assuming that the time stamp correspond to the time of read
        ts = np.linspace(ts[0]-dt, ts[-1], len(data))
    
    return strn,ts,data


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
        ret.set_trans(trans_mode.external, trans_value)  # TODO: save transMode of d1s when packing 
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
    w_tot = np.zeros(s0.qgrid.shape)
    label = None
    comments = ""
                
    for d1 in d1s:        
        # empty part of the data is nan
        idx = ~np.isnan(d1.data)
        # simple averaging
        #d_tot[idx] += d1.data[idx]
        #e_tot[idx] += d1.err[idx]
        c_tot[idx] += 1
        # average using 1/sigma as weight
        wt = 1/d1.err[idx]**2
        d_tot[idx] += wt*d1.data[idx]
        e_tot[idx] += d1.err[idx]**2*wt**2
        w_tot[idx] += wt
        
        idx1 = (np.ma.fix_invalid(d1.data, fill_value=-1)>d_max).data
        d_max[idx1] = d1.data[idx1]
        idx2 = (np.ma.fix_invalid(d1.data, fill_value=1e32)<d_min).data
        d_min[idx2] = d1.data[idx2]
            
        comments += d1.comments
        if label is None:
            label = d1.label
        else:
            label = common_name(label, d1.label)
    
    # simple averaging
    #s0.data[idx] /= c_tot[idx]
    #s0.err[idx] /= np.sqrt(c_tot[idx])
    # averaging by weight
    s0.data = d_tot/w_tot
    s0.err = np.sqrt(e_tot)/w_tot
    idx = (c_tot>1)
    s0.overlaps.append({'q_overlap': s0.qgrid[idx],
                        'raw_data1': d_max[idx],
                        'raw_data2': d_min[idx]})
    
    s0.label = label
    s0.comments = comments # .replace("# ", "## ")
    if save_merged:
        s0.save(s0.label+".dd", debug=debug)
        
    return s0

def proc_merge1d(args):
    """ utility function to perfrom azimuthal average and merge detectors
    """
    images,sn,nframes,starting_frame_no,debug,detectors,qgrid,reft,save_1d,save_merged,dtype = args
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
            dt.load_from_2D(images[det.extension][i], det.exp_para, qgrid, 
                            pre_process=det.pre_process, flat_cor=det.flat, mask=det.exp_para.mask,
                            save_ave=False, debug=debug, label=label, dtype=dtype)
            if det.dark is not None:
                dt.data -= det.dark
            dt.scale(sc[det.extension])
            ret[det.extension].append(dt)
    
        dm = merge_d1s([ret[det.extension][i] for det in detectors], detectors, save_merged, debug)
        ret['merged'].append(dm)
            
    if debug is True:
        print("processing completed: ", sn, starting_frame_no)

    return [sn, starting_frame_no, ret]
        
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
    
    def recalibrate(self, fn_std, energy, e_range=[5, 20], use_recalib=False,
                    det_type={"_SAXS": "Pilatus1M", "_WAXS2": "Pilatus1M"},
                    bkg={}, temp_file_location="/tmp"):
        """ fn_std should be a h5 file that contains AgBH pattern
            use the specified energy (keV) if the value is valid
            detector type
        """
        pxsize = 0.172e-3
        dstd = h5xs(fn_std, [self.detectors, self.qgrid]) 
        uname = os.getenv("USER")
        sn = dstd.samples[0]
        if energy>=e_range[0] and energy<=e_range[1]:
            wl = 2.*np.pi*1.973/energy
            for det in self.detectors:
                det.exp_para.wavelength = wl
        else:
            raise Exception(f"energy should be between {e_range[0]} and {e_range[1]} (keV): {energy}")
        
        for det in self.detectors:
            print(f"processing detector {det.extension} ...")    
            ep = det.exp_para
            data_file = f"{temp_file_location}/{uname}{det.extension}.cbf"

            with h5py.File(dstd.fn, "r") as fh5:
                img = fh5["%s/primary/data/%s" % (sn, dstd.det_name[det.extension])][0]

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
                if det.extension in bkg.keys():
                    img -= bkg[det.extension]
                fabio.cbfimage.CbfImage(data=img*(~dmask)).write(data_file)

            if use_recalib:
                # WARNING: pyFAI-recalib is obselete
                poni_file = f"/tmp/{uname}{det.extension}.poni"
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
                #cmd = ["pyFAI-recalib", "-i", poni_file, 
                #       "-c", "AgBh", "-r", "11", "--no-tilt", "--no-gui", "--no-interactive", data_file]
                cmd = ["pyFAI-calib", "-i", poni_file, 
                       "-c", "AgBh", "--no-tilt", "--no-gui", "--no-interactive", data_file]
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
            else:
                cmd = ["pyFAI-calib2",
                       "-D", det_type[det.extension],
                       "-w", f"{ep.wavelength:.4f}", "--dist", f"{pxsize*ep.Dd:.5f}",
                       "--poni1", f"{poni1:.6f}", "--poni2", f"{poni2:.6f}", "--no-tilt",
                       "-c", "AgBh", data_file]
                print("pyFAI-recalib is now obselete ...") 
                print("Run this interactively:")
                print(" ".join(cmd))
                fp = input("Then enter the path/name of the PONI file:")

                with open(fp, "r") as fh:
                    lines = {}
                    for _ in fh.read().split("\n"):
                        tl = _.split(":")
                        if len(tl)==2:
                            lines[tl[0]] = tl[1]

                print(f"  Original ::: bm_ctr_x = {ep.bm_ctr_x:.2f}, bm_ctr_y = {ep.bm_ctr_y:.2f}, ratioDw = {ep.ratioDw:.3f}")
                ep.ratioDw *= float(lines['Distance'])/(ep.Dd*pxsize)
                xc = float(lines['Poni2'])/pxsize
                yc = float(lines['Poni1'])/pxsize
                if ep.flip: ## can only handle flip=1 right now
                    ep.bm_ctr_x = yc
                    ep.bm_ctr_y = ep.ImageHeight-xc
                else: 
                    ep.bm_ctr_y = yc
                    ep.bm_ctr_x = xc
                print(f"   Revised ::: bm_ctr_x = {ep.bm_ctr_x:.2f}, bm_ctr_y = {ep.bm_ctr_y:.2f}, ratioDw = {ep.ratioDw:.3f}")
            ep.init_coordinates()

        self.save_detectors()

def find_field(fh5, fieldName, sname=None):
    tstream = {}
    if sname is None:
        samples = lsh5(fh5, top_only=True, silent=True)
    else:
        samples = [sname]
    for sn in samples:
        for stream in list(fh5[f"{sn}"]):
            if not 'data' in list(fh5[f"{sn}/{stream}"]):
                continue
            if fieldName in list(fh5[f"{sn}/{stream}/data"]):
                tstream[sn] = stream
                break
    if sname is None:
        return tstream
    
    return tstream[sname] 


def h5_file_access(method):
    @wraps(method)
    def inner(ref, *args, **kwargs):
        
        if not ref.fh5:
            file_open = False
            ref.fh5 = h5py.File(ref.fn, "r", swmr=True)
        else:
            file_open = True
            
        try:
            ret = method(ref, *args, **kwargs)
            if not file_open:  # leave the file open if it was initially open
                ref.fh5.close()
            return ret
        except:
            ref.fh5.close()
            raise
    
    return inner
    
class h5xs():
    """ Scattering data in transmission geometry
        Transmitted beam intensity can be set either from the water peak (sol), or from intensity monitor.
        Data processing can be done either in series, or in parallel. Serial processing can be forced.
        
        open h5 file only when necessary, using the h5_file_access decorator
        re-open the file for writing only when necessary
            save_attributes??
            save_detectors(), save_d1s()
        
    """    
    def __init__(self, fn, exp_setup=None, transField=transmitted_monitor, save_d1=True, 
                 have_raw_data=True, read_only=False):
        """ exp_setup: [detectors, qgrid]
            transField: the intensity monitor field packed by suitcase from databroker
            save_d1: save newly processed 1d data back to the h5 file
        """
        self.d0s = {}
        self.d1s = {}
        self.d2s = {}
        self.detectors = None
        self.samples = []
        self.attrs = {}
        # name of the dataset that contains transmitted beam intensity, e.g. em2_current1_mean_value
        self.transField = ''  
        self.transStream = {} 
        self.read_only = read_only
        self.writable = False   # this is the current state of the h5 file

        self.fn = fn
        self.fh5 = None
        #self.fh5 = h5py.File(fn, "r", swmr=True)   # file must exist; reopen for writing only when necessary
        self.save_d1 = save_d1

        self.setup(exp_setup, transField, have_raw_data)
        
    def explict_open_h5(self):
        if self.fh5:
            print(f"{self.hn} file is already open.")
        else:
            self.fh5 = h5py.File(self.fn, "r")
            
    def explicit_close_h5(self):
        if not self.fh5:
            print(f"{self.hn} file is already closed.")
        else:
            self.fh5.close()
        
    @h5_file_access  
    def setup(self, exp_setup=None, transField=transmitted_monitor, have_raw_data=True):
        if exp_setup is None:     # assume the h5 file will provide the detector config
            self.qgrid = self.read_detectors()
        else:
            self.detectors, self.qgrid = exp_setup
            if not self.read_only:
                self.save_detectors()
        self.list_samples(quiet=True)

        if have_raw_data:
            # find out what are the fields corresponding to the 2D detectors
            # at LiX there are two possibilities; assume all samples have have data stored in the same fileds
            sn = self.samples[0]
            streams = list(self.fh5[sn].keys())
            data_fields = {}
            for stnm in streams:
                if 'data' in self.fh5[f"{sn}/{stnm}"].keys(): 
                    for tf in list(self.fh5[f"{sn}/{stnm}/data"]):
                        data_fields[tf] = stnm
        
            self.det_name = None
            # these are the detectors that are present in the data
            d_dn = [d.extension for d in self.detectors]
            for det_name in det_names:
                dn = copy.copy(det_name)
                for k,v in det_name.items():
                    if not k in d_dn or not v in data_fields.keys():
                        del dn[k]
                if len(dn.keys())>0:
                    self.det_name = dn
                    break

            if self.det_name is None:
                print('fields in the h5 file: ', data_fields)
                raise Exception("Could not find the data corresponding to the detectors.")

            # transStream is more complicated
            # different samples may store the data in different streams 
            if transField=='': 
                if 'trans' in self.fh5.attrs:
                    tf = self.fh5.attrs['trans'].split(',')
                    # transMove, transField, transStream 
                    # but transStream is not always recorded
                    v, self.transField = tf[:2]
                    self.transMode = trans_mode(int(v))
                    self.transStream = find_field(self.fh5, self.transField)
                    return
                else:
                    self.transMode = trans_mode.from_waxs
                    self.transField = ''
                    self.transStream = {}
            else:
                try:
                    self.transStream = find_field(self.fh5, transField)
                except:
                    print("invalid field for transmitted intensity: ", transField)
                    raise Exception()
                self.transField = transField
                self.transMode = trans_mode.external

            if not self.read_only:
                self.enable_write(True)
                self.fh5.attrs['trans'] = ','.join([str(self.transMode.value), self.transField])  #elf.transStream])
                self.enable_write(False)
                
    def enable_write(self, writable):
        if self.read_only:
            if not writable:
                return
            raise Exception(f"attempting to enable writing to a read-only h5 file ...")
        if self.writable == writable:
            return
        self.fh5.close()
        if writable:
            self.fh5 = h5py.File(self.fn, "r+")
            #self.fh5.swmr_mode = True
        else:
            self.fh5 = h5py.File(self.fn, "r", swmr=True)
        self.writable = writable
    
    def det(self, ext):
        for d in self.detectors:
            if d.extension==ext:
                return d
        return None
    
    @h5_file_access  
    def save_detectors(self):
        if self.read_only:
            return
        self.enable_write(True)
        dets_attr = [det.pack_dict() for det in self.detectors]
        self.fh5.attrs['detectors'] = json.dumps(dets_attr)
        self.fh5.attrs['qgrid'] = list(self.qgrid)
        self.enable_write(False)
    
    @h5_file_access  
    def read_detectors(self):
        dets_attr = self.fh5.attrs['detectors']
        qgrid = self.fh5.attrs['qgrid']
        self.detectors = [create_det_from_attrs(attrs) for attrs in json.loads(dets_attr)]  
        return np.asarray(qgrid)

    @h5_file_access
    def get_h5_attr(self, grp, attr_name):
        return self.fh5[grp].attrs[attr_name]

    @h5_file_access
    def set_h5_attr(self, grp, attr_name, value):
        self.enable_write(True)
        self.fh5[grp].attrs[attr_name] = value
        self.enable_write(False)
        
    @h5_file_access
    def dshape(self, dsname, data_type="data", sn=None):
        assert (data_type in ["data", "timestamps"])
        if sn is None:
            sn = self.samples[0]
        strm = find_field(self.fh5, dsname, sn)
        return self.fh5[f"{sn}/{strm}/{data_type}/{dsname}"].shape
    
    @h5_file_access
    def dset(self, dsname, data_type="data", sn=None, get_path=False, return_reference=True):
        assert (data_type in ["data", "timestamps"])
        if sn is None:
            sn = self.samples[0]
        strm = find_field(self.fh5, dsname, sn)
        if get_path:
            return f"{sn}/{strm}/{data_type}/{dsname}"
        if return_reference:
            return self.fh5[f"{sn}/{strm}/{data_type}/{dsname}"]   # just return a reference  
        else:
            return self.fh5[f"{sn}/{strm}/{data_type}/{dsname}"][...]
    
    @h5_file_access  
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
    
    @h5_file_access  
    def header(self, sn):
        if not sn in self.samples:
            raise Exception(f"{sn} is not a valid sample.")
        if "start" in self.fh5[sn].attrs:
            return json.loads(self.fh5[sn].attrs['start'])
        return None
    
    @h5_file_access  
    def list_samples(self, quiet=False):
        self.samples = lsh5(self.fh5, top_only=True, silent=True)
        if not quiet:
            print(self.samples)
    
    @h5_file_access  
    def verify_frn(self, sn, frn, flatten=False):
        """ simply translate between a scaler index and a multi-dimensional index based on scan shape
            this became more complicated when areadetector saves data as hdf
            sshape and dshape are no longer the same
            not taking care of snaking here
        """
        header = self.header(sn)
        if 'shape' in header.keys():
            sshape = header['shape']
            #snaking = header['snaking']
        elif 'num_points' in header.keys():
            sshape = [header["num_points"]]
            #snaking = False
        else:
            raise Exception("don't kno how to handler the header", header)
        dshape = self.fh5[f"{sn}/primary/data/{list(self.det_name.values())[0]}"].shape[:-2]
    
        if frn is None:
            frn = 0
        if hasattr(frn, '__iter__'): # tuple or list or np array
            if len(frn)==1:
                frn = frn[0]
            elif len(frn)!=len(sshape):
                raise Exception(f"invalid frame number {frn}, must contain {len(sshape)} element(s).")
                
        if isinstance(frn, int): # translate to the right shape
            if flatten:
                return frn
            idx = []
            for i in reversed(range(len(dshape))):
                idx = [frn%dshape[i]]+idx
                frn = int(frn/dshape[i])
            frn = idx
        if flatten:
            frn1 = 0 
            gs = 1 # fastest axis, increasing the index by 1 = next frame
            for i in reversed(range(len(frn))):
                frn1+=gs*frn[i]
                gs*=sshape[i]
            return frn1
        return frn
    
    def get_d1(self, sn=None, group="merged", frn=None):
        if sn is None:
            sn = self.samples[0]
        if not group in self.d1s[sn].keys():
            raise Exception(f"1d data do not exist under {group}.")
        frn = self.verify_frn(sn, frn, flatten=True)
        return self.d1s[sn][group][frn]    
            
    @h5_file_access  
    def get_d2(self, sn=None, det_ext=None, frn=None, dtype=None, detectors=None, client=None):
        if sn is None:
            sn = self.samples[0]
        d2s = {}
        if detectors is None:
            detectors = self.detectors
            
        if frn=="average" and client is None:
            raise Exception("need a dask client for calculating the average.")
            
        for det in detectors:
            try:
                dset = self.fh5[f"{sn}/primary/data/{self.det_name[det.extension]}"]
            except:
                continue
            if frn=="average":
                if not USE_DASK:
                    raise Exception("Dask is not available for averaging frames ...")
                imgs = da.array(dset)
                fut = da.average(imgs, axis=tuple(range(len(dset.shape)-2)))
                davg = fut.compute(client=client)
                d2 = Data2d(davg, exp=det.exp_para, dtype=dtype)
            else:
                frn = self.verify_frn(sn, frn)
                d2 = Data2d(dset[tuple(frn)], exp=det.exp_para, dtype=dtype)
            d2.md["frame #"] = frn 
            d2s[det.extension] = d2

        if not det_ext:
            return d2s
        else:
            return d2s[det_ext]
                
    def check_bm_center(self, sn=None, det_ext='_SAXS', frn=0, 
                        qs=0.005, qe=0.05, qn=100, Ns=9, **kwargs):
        """ this function compares the beam intensity on both sides of the beam center,
            and advise if the beam center as defined in the detector configuration is incorrect
            dividing data into 4*Ns slices, show data in horizontal and vertical cuts
        """
        i = 0
        d2 = self.get_d2(sn=sn, det_ext=det_ext, frn=frn, **kwargs)
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
        
    def show_data(self, sn=None, det_ext=None, frn=None, ax=None, fig=None, aspect=1,
                  logScale=True, showMask=False, mask_alpha=0.1, clim='auto', ch_thresh=15,
                  showRef=["AgBH", "r:"], cmap=None, detectors=None, dtype=None, **kwargs):
        """ display frame #frn of the data for sample sn and frame number frn
            if not specified, take the first sample and/or first frame

            if det_ext is None, show data from all detectors
            otherwise show the data that correspond to the specified detector extension, e.g. "_SAXS"

            when clim is 'auto', check the distribution/histogram of intensity, set limits 
                to include only bins with number of samples greater than ch_thresh
        """
        if detectors is None:
            detectors = self.detectors
        d2s = self.get_d2(sn=sn, det_ext=det_ext, frn=frn, dtype=dtype, detectors=detectors, **kwargs)
        show_data(d2s, ax=ax, fig=fig, aspect=aspect,
                  logScale=logScale, showMask=showMask, mask_alpha=mask_alpha, 
                  clim=clim, showRef=showRef, cmap=cmap, dtype=dtype, **kwargs)
        return d2s
    
    def show_data_qxy(self, sn=None, frn=None, ax=None, dq=0.01, bkg=None,
                      logScale=True, useMask=True, clim=(0.1,14000), 
                      aspect=1, cmap=None, detectors=None, dtype=None, colorbar=False, **kwargs):
        """ display frame #frn of the data under det for sample sn
            det is a list of detectors, or a string, data file extension
        """
        d2s = self.get_d2(sn=sn, frn=frn, dtype=dtype, **kwargs)
        if detectors is None:
            detectors = self.detectors
        return show_data_qxy(d2s, detectors, ax=ax, dq=dq, bkg=bkg,
                             logScale=logScale, useMask=useMask, clim=clim, 
                             aspect=aspect, cmap=cmap, colorbar=colorbar, **kwargs)

    def show_data_qphi(self, sn=None, frn=None, ax=None, Nq=200, Nphi=60,
                       apply_symmetry=False, fill_gap=False, sc_factor=None,
                       logScale=True, useMask=True, clim=(0.1,14000), showRef=True, bkg=None,
                       aspect="auto", cmap=None, detectors=None, dtype=None, colorbar=False, **kwargs):
        d2s = self.get_d2(sn=sn, frn=frn, dtype=dtype, **kwargs)
        if detectors is None:
            detectors = self.detectors
        return show_data_qphi(d2s, detectors, ax=ax, Nq=200, Nphi=60,
                              apply_symmetry=apply_symmetry, fill_gap=fill_gap, 
                              sc_factor=sc_factor, bkg=bkg,
                              logScale=logScale, useMask=useMask, clim=clim, 
                              aspect=aspect, cmap=cmap, colorbar=colorbar, **kwargs)

    @h5_file_access  
    def get_ts(self, sn, exp, trigger="sol"):
        if trigger=="sol" or trigger is None:
            dn = list(self.det_name.values())[0]
            # expect a finite but minimal offset in time since all come from the same IOC server
            ts0 = self.fh5[f'{sn}/primary/timestamps/{dn}'][...].flatten()
            dshape = self.fh5[f"{sn}/primary/data/{dn}"].shape[0]   # length of the time sequence
            if len(ts0)==1: # multiple exposures, single trigger, as in HT measurements
                ts0 = ts0[0]+np.arange(dshape)*exp    
        elif trigger in self.fh5[f'{sn}/primary/timestamps'].keys():
            ts0 = get_det_timestamps(self.fh5[f'{sn}/primary/'], trigger, exp)
        else:
            raise Exception(f"timestamp data for {trigger} cannot be found.")

        return ts0
    
    @h5_file_access  
    def get_mon(self, sn=None, trigger=None, gf_sigma=2, exp=1, check_size=0,
                force_synch='auto', force_synch_trig=0, extend_mon_stream=True,
                plot_trigger=False, **kwargs): 
        """ calculate the monitor counts for each data point
            1. if the monitors are read together with the detectors 
            2. if the monitors are used asynchronously, monitor values would need to be integated 
               based on timestamps; a trigger must be provided 
                2a: in fly scans, this should be a motor name
                2b: for solution scattering, use "sol" as trigger

            the timestamps on the trigger and em1/em2 may not be synced, dt0 provides a correction 
            to the trigger timestamp; 
            
            if plot_trigger=True and a single sample is named as sn, generate a plot for verification

            timestamps on em1 appear to be way off, ntpd not running
            if force_synch is non-zero, use first em2 timestamp + force_synch as start of em1
            in principle can be synched by the shutter opening seen by the monitors when the scan starts
            
            in some cases, the monitor stream appears to end before data collection is complete
            if extend_mon_stream=True, repeat the last mon data point until the timestamp covers
            the range of the trigger
        """
        if sn is None or isinstance(sn, list):
            if plot_trigger:
                plot_trigger = False           # plot for a single sample only
                print("Disabling plot_trigger since not a single sample name is specified.")
            samples = self.samples
        else:
            samples = [sn]
           
        for sn in samples:
            if "pilatus" in self.header(sn).keys():
                md = self.header(sn)['pilatus']
                if 'exposure_time' in md.keys():
                    exp=md['exposure_time']

            # expect trans and incid monitor data to be in the same stream
            strn,ts2,trans_data = get_monitor_counts(self.fh5[sn], transmitted_monitor)
            strn,ts1,incid_data = get_monitor_counts(self.fh5[sn], incident_monitor)
                
            if strn=="primary":
                print("monitors are used as detectors.")
                trans_data0 = trans_data
                incid_data0 = incid_data
            else:
                # synch em1 time stamps to em2, LiX-specific 
                synch_mon(ts2, trans_data, ts1, incid_data, dt=force_synch)
                ts0 = self.get_ts(sn, exp, trigger)

                try:
                    trans_data0 = integrate_mon(trans_data, ts2, ts0+force_synch_trig, exp, extend_mon_stream, **kwargs)
                    incid_data0 = integrate_mon(incid_data, ts1, ts0+force_synch_trig, exp, extend_mon_stream, **kwargs)   
                except:
                    t0 = np.min(ts2)
                    print(f"time series likely misaligned:")
                    print(f"trans mon: {np.min(ts2)-t0} ~ {np.max(ts2)-t0}")
                    print(f"incid mon: {np.min(ts1)-t0} ~ {np.max(ts1)-t0}")
                    print(f"detector: start={ts0-t0}, force_synch={force_synch_trig}, exp = {exp}")
                    ts1 += ts2[0]-ts1[0]
                    print(f"using force_synch=0, incid mon: {np.min(ts1)-t0} ~ {np.max(ts1)-t0}")
                    incid_data0 = integrate_mon(incid_data, ts1, ts0+force_synch_trig, exp, extend_mon_stream)

                if plot_trigger:
                    plt.figure()
                    plt.plot(ts2, trans_data/np.max(trans_data))
                    plt.plot(ts0+force_synch_trig, trans_data0/np.max(trans_data), "o")
                    plt.plot(ts1, incid_data/np.max(incid_data))
                    plt.plot(ts0+force_synch_trig, incid_data0/np.max(incid_data), "o")
                    
            if not hasattr(self, "d0s"):
                self.d0s = {}
            if not sn in self.d0s.keys():
                self.d0s[sn] = {}

            # make sure that the monitor counts have the right number of data points
            if check_size>0: 
                check_size -= len(trans_data0)
            self.d0s[sn]["transmitted"] = np.pad(trans_data0, (0,check_size), constant_values=np.nan)
            self.d0s[sn]["incident"] = np.pad(incid_data0, (0,check_size), constant_values=np.nan)
            transmission = trans_data0/incid_data0
            #transmission /= np.nanmean(transmission)   # this can cause problems when the beam is off during part of the scan
            self.d0s[sn]["transmission"] = np.pad(transmission, (0,check_size), constant_values=np.nan)
            
            
    def set_trans(self, transMode, sn=None, trigger=None, gf_sigma=2, dt0=-133.8, exp=1, plot_trigger=False, **kwargs): 
        """ set the transmission values for the merged data
            the trans values directly from the raw data (water peak intensity or monitor counts)
            but this value is changed if the data is scaled
            the trans value for all processed 1d data are saved as attrubutes of the dataset

            when em2 is used as a monitor, a trigger must be provided to find the corresponding monitor value
            in fly scans, this should be a motor name
            for solution scattering, use "sol" as trigger

            the timestamps on the trigger and em2 may not be synced, dt0 provides a correction to the trigger timestamp
            if plot_trigger=True and sn is not None, will generate a plot for verification
            
            dt0 for sol trigger should be finite but minimal, ~0.05?

        """
        assert(isinstance(transMode, trans_mode))
        self.transMode = transMode
        if self.transMode==None:
            raise Exception("a valid transmited intensity mode must be specified.")

        if sn is None:
            if plot_trigger:
                plot_trigger = False           # plot for a single sample only
                print("Disabling plot_trigger since no sample name is specified.")
            samples = self.samples
        else:
            samples = [sn]
        for s in samples:
            if self.transMode==trans_mode.external: # transField should be set/validated already
                assert(self.transField!='')
                try:
                    trans_data = self.d0s[s]["transmitted"]
                except:
                    print("attempting to run get_mon() first ...")
                    self.get_mon(sn=s, trigger=trigger)
                    trans_data = self.d0s[s]["transmitted"]

            # these are the datasets that needs updated trans values
            # also need to rescale merged data, in case it's been scale, e.g. during normalization
            grps = list(set(self.d1s[s].keys()) & set(det_model.keys()))
            grp0 = grps[0]
            if "merged" in self.d1s[s].keys():
                grps += ["merged"]
            if len(grps)==0:
                return

            t_values = []
            for i in range(len(self.d1s[s][grp0])):
                if "merged" in grps:
                    idx = (self.d1s[s][grp0][i].data>0)
                    sc0 = np.sum(self.d1s[s][grp0][i].data[idx])/np.sum(self.d1s[s]["merged"][i].data[idx])
                    self.d1s[s]["merged"][i].scale(sc0)
                for grp in grps:
                    if self.transMode==trans_mode.external:
                        self.d1s[s][grp][i].set_trans(self.transMode, trans_data[i], **kwargs)
                    else:
                        self.d1s[s][grp][i].set_trans(self.transMode, **kwargs)
                t_values.append(self.d1s[s][grp0][i].trans)
            if 'averaged' in self.d1s[s].keys():
                self.d1s[s]['averaged'].trans = np.average(t_values)
    
    @h5_file_access  
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
            self.d0s[sn] = {}    # these are attributes extracted from each data point
            self.d1s[sn] = {}    # 1d scattering data, of the type data1d
            self.d2s[sn] = {}    # 2d data, of the type MatrixWithCorrds
            self.attrs[sn] = {}
        grp = fh5[sn+'/processed']
        for k in list(grp.attrs.keys()):
            self.attrs[sn][k] = grp.attrs[k]   
        # initially only handles 1d data
        for k in lsh5(grp, top_only=True, silent=True):
            if 'trans' in grp[k].attrs.keys():
                tvs = grp[k].attrs['trans']
            else:
                tvs = 0
            self.d1s[sn][k] = unpack_d1(grp[k], self.qgrid, sn+k, tvs)   
            

    @h5_file_access  
    def save_d1s(self, sn=None, debug=False):
        """
        save the 1d data in memory to the hdf5 file 
        processed data go under the group sample_name/processed
        assume that the shape of the data is unchanged
        """
        
        if self.read_only:
            print("h5 file is read-only ...")
            return
        if self.save_d1 is False:
            print("requested to save_d1s() but h5xs.save_d1 is False.")
            return
        if sn==None:
            self.list_samples(quiet=True)
            for sn in self.samples:
                self.save_d1s(sn)
        
        self.enable_write(True)        
        if "processed" not in list(lsh5(self.fh5[sn], top_only=True, silent=True)):
            grp = self.fh5[sn].create_group("processed")
        else:
            grp = self.fh5[sn+'/processed']
            g0 = lsh5(grp, top_only=True, silent=True)[0]
            if grp[g0][0].shape[1]!=len(self.qgrid): # if grp[g0].value[0].shape[1]!=len(self.qgrid):
                # new size for the data
                del self.fh5[sn+'/processed']
                grp = self.fh5[sn].create_group("processed")
        
        # these attributes are not necessarily available when save_d1s() is called
        if sn in list(self.attrs.keys()):
            for k in list(self.attrs[sn].keys()):
                grp.attrs[k] = self.attrs[sn][k]
                if debug is True:
                    print(f"writing attribute to {sn}: {k}")

        ds_names = lsh5(grp, top_only=True, silent=True)
        for k in list(self.d1s[sn].keys()):
            data,tvs = pack_d1(self.d1s[sn][k])
            if debug is True:
                print(f"writing attribute to {sn}: {k}")
            if k not in ds_names:
                grp.create_dataset(k, data=data)
            else:
                grp[k][...] = data   

            # save trans values for processed data
            # before 1d data merge, the trans value should be 0              
            # on the other hand there could be data collected with the beam off, therefore trans=0
            if (np.asarray(tvs)>0).any(): 
                grp[k].attrs['trans'] = tvs
                
        self.enable_write(False)

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
                if selection is None:
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
            print("done, time elapsed: %.2f sec" % (t2-t1))
            
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
                
    @h5_file_access  
    def load_data(self, update_only=False, detectors=None,
           reft=-1, save_1d=False, save_merged=False, debug=False, N=8, max_c_size=0, dtype=None):
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
                n_total_frames = s[-3]  # fast axis
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
                        job = pool.map_async(proc_merge1d, [(images, sn, nframes, i*c_size, debug,
                                                             detectors, self.qgrid, reft, save_1d, save_merged, dtype)])
                        jobs.append(job)
                    else: # serial processing
                        [sn, fr1, data] = proc_merge1d((images, sn, nframes, i*c_size, debug, 
                                                        detectors, self.qgrid, reft, save_1d, save_merged, dtype)) 
                        results[sn][fr1] = data                
                else: # len(s)==4
                    for j in range(s[0]):  # slow axis
                        images = {}
                        for det in detectors:
                            gn = f'{self.det_name[det.extension]}'
                            images[det.extension] = dset[gn][j, i*c_size:i*c_size+nframes]
                        if N>1: # multi-processing, need to keep track of total number of active processes
                            job = pool.map_async(proc_merge1d, [(images, sn, nframes, i*c_size+j*s[1], debug,
                                                                 detectors, self.qgrid, reft, save_1d, save_merged, dtype)])
                            jobs.append(job)
                        else: # serial processing
                            [sn, fr1, data] = proc_merge1d((images, sn, nframes, i*c_size+j*s[1], debug, 
                                                            detectors, self.qgrid, reft, save_1d, save_merged, dtype)) 
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

