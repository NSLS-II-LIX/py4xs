# need to have a more uniform method to exchange (pack/unpack) 1D and 2D PROCESSED data with hdf5
# type of data: Data1d, MatrixWithCoordinates (not just simple numpy arrays)
import pylab as plt
import h5py
import numpy as np
import time, datetime
import copy
import json,pickle
import multiprocessing as mp

from py4xs.slnxs import Data1d, average, filter_by_similarity, trans_mode
from py4xs.utils import common_name
from py4xs.detector_config import create_det_from_attrs
from py4xs.local import det_names    # e.g. "_SAXS": "pil1M_image"
from py4xs.data2d import Data2d, Axes2dPlot
from itertools import combinations

from scipy.linalg import svd
from scipy.ndimage.filters import gaussian_filter

def lsh5(hd, prefix='', top_only=False, silent=False):
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
            print(list(hd[k].attrs.items()))
            lsh5(hd[k], prefix+"=")

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
            self.fh5 = h5py.File(self.fn, "r")   # file must exist
            self.qgrid = self.read_detectors()
        else:
            self.fh5 = h5py.File(self.fn, "w")   # new file
            self.detectors, self.qgrid = exp_setup
            self.save_detectors()
        self.fh5.close()
        
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
        
        
class h5xs():
    """ Scattering data in transmission geometry
        Transmitted beam intensity can be set either from the water peak (sol), or from intensity monitor.
        Data processing can be done either in series, or in parallel. Serial processing can be forced.
        
    """
    d1s = {}
    detectors = None
    samples = []
    attrs = {}
    # name of the dataset that contains transmitted beam intensity, e.g. em2_current1_mean_value
    transField = None  
    
    def __init__(self, fn, exp_setup=None, transField='', save_d1=True):
        """ exp_setup: [detectors, qgrid]
            transField: the intensity monitor field packed by suitcase from databroker
            save_d1: save newly processed 1d data back to the h5 file
        """
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
        data_fields = list(self.fh5[self.samples[0]+'/primary/data'])
        self.det_name = None
        # these are the detectors that are present in the data
        d_dn = [d.extension for d in self.detectors]
        for det_name in det_names:
            for k in set(det_name.keys()).difference(d_dn):
                del det_name[k]
            if set(det_name.values()).issubset(data_fields):
                self.det_name = det_name
                break
        if self.det_name is None:
            print('fields in the h5 file: ', data_fields)
            raise Exception("Could not find the data corresponding to the detectors.")
        if transField=='':
            if 'trans' in self.fh5.attrs:
                [v, self.transField] = self.fh5.attrs['trans'].split(',')
                self.transMode = trans_mode(int(v))
                return
            else:
                self.transMode = trans_mode.from_waxs
                self.transField = ''
        elif transField not in data_fields:
            print("invalid filed for transmitted intensity: ", transField)
            raise Exception()
        else:
            self.transField = transField
            self.transMode = trans_mode.external
        self.fh5.attrs['trans'] = ','.join([str(self.transMode.value), self.transField])
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

    def list_samples(self, quiet=False):
        self.samples = lsh5(self.fh5, top_only=True, silent=True)
        if not quiet:
            print(self.samples)
    
    def show_data(self, sn=None, det='_SAXS', frn=0, 
                  logScale=True, showMask=False, clim=(0.1,14000), showRef=True, cmap=None):
        """ display frame #frn of the data under det for sample sn
        """
        if sn is None:
            sn = self.samples[0]
        dset = self.fh5["%s/primary/data/%s" % (sn, self.det_name[det])]
        di = np.arange(len(self.detectors))[np.asarray([d.extension for d in self.detectors])==det][0]
        exp = self.detectors[di].exp_para
        d2 = Data2d(filename=None, im=dset[frn], exp=exp)
        
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
    
    def set_trans(self, sn=None, transMode=None):
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
            # these are the datasets that needs updated trans values
            if 'merged' not in self.d1s[s].keys():
                continue
            t_values = []
            for i in range(len(self.d1s[s]['merged'])):
                if self.transMode==trans_mode.external:
                    self.d1s[s]['merged'][i].set_trans(self.fh5[s+'/primary/data/'+self.transField][i], 
                                                        transMode=self.transMode)
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
            
    def plot_d1s(self, sn, ax=None, offset=1.5, 
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
            self.d1s[sn]['subtracted'].plot(ax=ax)
            if show_subtraction:
                self.d1s[sn]['averaged'].plot(ax=ax)
                self.d1b[sn].plot(ax=ax)
        else:
            sc = 1
            for i,d1 in enumerate(self.d1s[sn]['merged']):
                if self.attrs[sn]['selected'][i]:
                    d1.plot(ax=ax, scale=sc)
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

    def export_d1s(self, samples=None, path="", save_subtracted=True, debug=False):
        """ if path is used, be sure that it ends with '/'
        """
        if samples is None:
            samples = list(self.buffer_list.keys())
        elif isinstance(samples, str):
            samples = [samples]

        if debug is True:
            print("start processing: export_txt()")
        for sn in samples:
            if save_subtracted:
                if 'subtracted' not in self.d1s[sn].keys():
                    print("subtracted data not available.")
                    return
                self.d1s[sn]['subtracted'].save("%s%s_%c.dat"%(path,sn,'s'), debug=debug)
            else:
                if 'merged' not in self.d1s[sn].keys():
                    print("1d data not available.")
                    return
                for i in range(len(self.d1s[sn]['merged'])):
                    self.d1s[sn]['merged'][i].save("%s%s_%d%c.dat"%(path,sn,i,'m'), debug=debug)                    
                
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
            if len(s)==3:
                n_total_frames = s[0]
            elif len(s)==4:
                n_total_frames = s[1]
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
            for i in range(Np):
                if i==Np-1:
                    nframes = n_total_frames - c_size*(Np-1)
                else:
                    nframes = c_size
                    
                images = {}
                if detectors is None:
                    detectors = self.detectors
                for det in detectors:
                    if len(s)==3:
                        images[det.extension] = dset['%s' % self.det_name[det.extension]][i*c_size:i*c_size+nframes]
                    else:
                        images[det.extension] = dset['%s' % self.det_name[det.extension]][0][i*c_size:i*c_size+nframes]
                if N>1: # multi-processing, need to keep track of total number of active processes                    
                    job = pool.map_async(proc_d1merge, [(images, sn, nframes, i*c_size, debug,
                                                         detectors, self.qgrid, reft, save_1d, save_merged)])
                    jobs.append(job)
                else: # serial processing
                    [sn, fr1, data] = proc_d1merge((images, sn, nframes, i*c_size, debug, 
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
                ti = fh5["%s/primary/data/%s" % (sn, self.det_name[det.extension])].value
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
    dbuf = None
    updating = False   # this is set to True when add_data() is active
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def add_data(self, ):
        """ watch the given path
            update self.d1s when new files are found
            save to HDF when the scan is done
        """
        pass
    
    def process_sample_name(self, sn, debug=False):
        #fh5 = h5py.File(self.fn, "r+")
        fh5 = self.fh5
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        if sn==None:
            sn = self.samples[0]
        elif sn not in self.samples:
            raise Exception(sn, "not in the sample list.")
        
        return fh5,sn 
        
    def load_d1s(self):
        super().load_d1s(self.samples[0])
        # might need to update meta data??
        
    def normalize_int(self, ref_trans=-1):
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
        
    def process(self, update_only=False,
                reft=-1, save_1d=False, save_merged=False, 
                filter_data=False, debug=False, N=8, max_c_size=0):
        """ load data from 2D images, merge, then set transmitted beam intensity
        """

        self.load_data(update_only=update_only, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N, max_c_size=max_c_size)
        self.set_trans(transMode=trans_mode.from_waxs)
        self.normalize_int()

    def subtract_buffer_SVD(self, excluded_frames_list, sn=None, sc_factor=0.995,
                            gaussian_filter_width=None,
                            Nc=5, poly_order=8,
                            plot_fit=False, ax=None, debug=False):
        """ perform SVD background subtraction, use Nc eigenvalues 
            poly_order: order of polynomial fit to each eigenvalue
            gaussian_filter width: sigma value for the filter, e.g. 1, or (0.5, 3)
        """
        fh5,sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()
            
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
        for i in range(Nc):
            Uf.append(np.poly1d(np.polyfit(bkg_frns, U[:,i], poly_order)))
        Ub = np.vstack([f(all_frns) for f in Uf]).T
        dd2c = np.dot(np.dot(Ub, np.diag(s[:Nc])), Vh[:Nc,:]).T

        if plot_fit:
            if ax is None:
                plt.figure()
                ax = plt.gca()
            for i in reversed(range(Nc)):
                ax.plot(bkg_frns, np.sqrt(s[i])*U[:,i], '.') #s[i]*U[:,i])
            for i in reversed(range(Nc)):
                ax.plot(all_frns, np.sqrt(s[i])*Uf[i](all_frns)) #s[i]*U[:,i])    

        self.attrs[sn]['sc_factor'] = sc_factor
        self.attrs[sn]['svd excluded frames'] = excluded_frames_list
        self.attrs[sn]['svd parameters'] = [Nc, poly_order]
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
            
    def plot_data(self, sn=None, i_minq=0.02, i_maxq=0.05, flowrate=0, plot_merged=False,
                  ymin=-1, ymax=-1, offset=0, uv_scale=1, showFWHM=False, 
                  calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True,
                  plot2d=True, logScale=True, clim=[1.e-3, 10.],
                  export_txt=False, debug=False, fig_w=8, fig_h1=2, fig_h2=3.5):
        """ plot "merged" if no "subtracted" present
            export_txt: export the scattering-intensity-based chromatogram
        """
        
        if plot2d:
            plt.figure(figsize=(fig_w, fig_h1+fig_h2))
            plt.subplot(211)
        else:
            plt.figure(figsize=(fig_w, fig_h2))

        fh5,sn = self.process_sample_name(sn, debug=debug)

        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax3 = ax1.twinx()
        
        if 'subtracted' in self.d1s[sn].keys() and plot_merged==False:
            dkey = 'subtracted'
        elif 'merged' in self.d1s[sn].keys():
            if plot_merged==False:
                print("subtracted data not available. plotting merged data instead ...")
            dkey = 'merged'
        else:
            raise Exception("processed data not present.")
            
        data = self.d1s[sn][dkey]
        #qgrid = data[0].qgrid
        ts = fh5[sn+'/primary/time'].value
        #idx = (qgrid>i_minq) & (qgrid<i_maxq) 
        idx = (self.qgrid>i_minq) & (self.qgrid<i_maxq) 
        data_t = []
        data_i = []
        data_rg = []
        ds = []
        frame_n = 0
        for i in range(len(data)):
            ti = (ts[i]-ts[0])/60
            if flowrate>0:
                ti*=flowrate

            ii = data[i].data[idx].sum()
            ds.append(data[i].data)

            if ii>thresh and calc_Rg and dkey=='subtracted':
                i0,rg = dt.plot_Guinier(qs, qe, fix_qe=fix_qe, no_plot=True)
                #print("frame # %d: i0=%.2g, rg=%.2f" % (frame_n,i0,rg))
            else:
                rg = 0

            data_t.append(ti)
            data_i.append(ii)
            data_rg.append(rg)
            frame_n += 1

        if ymin == -1:
            ymin = np.min(data_i)
        if ymax ==-1:
            ymax = np.max(data_i)

        if export_txt:
            # export the scattering-intensity-based chromatogram
            np.savetxt(sn+'.chrome', np.vstack((data_t, data_i)).T, "%12.3f")
            
        ax1.plot(data_i, 'b-')
        ax1.set_xlabel("frame #")
        ax1.set_xlim((0,len(data_i)))
        ax1.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
        ax1.set_ylabel("intensity")

        # read HPLC data directly from HDF5
        hplc_grp = fh5[sn+"/hplc/data"]
        fields = lsh5(fh5[sn+'/hplc/data'], top_only=True, silent=True)
        dc = []
        for fd in fields:
            dc = fh5[sn+'/hplc/data/'+fd].value.T
            ax2.plot(np.asarray(dc[0])+offset,
                     ymin+dc[1]/np.max(dc[1])*(ymax-ymin)*uv_scale, label=fd)
            #ax2.set_ylim(0, np.max(dc[0][2]))

        if flowrate>0:
            ax2.set_xlabel("volume (mL)")
        else:
            ax2.set_xlabel("time (minutes)")
        ax2.plot(data_t, data_i, 'bo', label='x-ray ROI')
        ax2.set_xlim((data_t[0],data_t[-1]))

        if showFWHM:
            half_max=(np.amax(data_i)-np.amin(data_i))/2 + np.amin(data_i)
            s = splrep(data_t, data_i - half_max)
            roots = sproot(s)
            fwhm = abs(roots[1]-roots[0])
            print(roots[1],roots[0],half_max)
            if flowrate>0:
                print("X-ray cell FWHMH =", fwhm, "ml")
            else:
                print("X-ray cell FWHMH =", fwhm, "min")
            ax2.plot([roots[0], roots[1]],[half_max, half_max],"k-|")

        if calc_Rg and dkey=='subtracted':
            data_rg = np.asarray(data_rg)
            max_rg = np.max(data_rg)
            data_rg[data_rg==0] = np.nan
            ax3.plot(data_rg, 'r.', label='rg')
            ax3.set_xlim((0,len(data_rg)))
            ax3.set_ylim((0, max_rg*1.05))
            ax3.set_ylabel("Rg")

        leg = ax2.legend(loc='upper left', fontsize=9, frameon=False)
        leg = ax3.legend(loc='center left', fontsize=9, frameon=False)

        if plot2d:
            plt.subplots_adjust(bottom=0.)
            plt.subplot(212)
            plt.subplots_adjust(top=1.)
            ax = plt.gca()
            ax.tick_params(axis='x', top=True)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            #ax3 = ax1.twinx()
            d2 = np.vstack(ds).T + clim[0]/2
            ext = [0, len(data), self.qgrid[-1], self.qgrid[0]]
            asp = len(data)/self.qgrid[-1]/(fig_w/fig_h1)
            if logScale:
                plt.imshow(np.log(d2), extent=ext, aspect=asp) 
                plt.clim(np.log(clim))
            else:
                plt.imshow(d2, extent=ext, aspect=asp) 
                plt.clim(clim)
            #plt.xlabel('frame #')
            plt.ylabel('q')
            #ax3.set_xlabel('frame #')
            plt.tight_layout()

        plt.tight_layout()
        plt.show()
        
    def bin_subtracted_frames(self, sn=None, first_frame=0, last_frame=-1, 
                              plot_data=True, fig=None, qmax=0.5, qs=0.01,
                              save_data=False, path="", debug=False): 
        """ this is typically used after running subtract_buffer_SVD()
            if path is used, be sure that it ends with '/'
        """
        fh5,sn = self.process_sample_name(sn, debug=debug)
        if last_frame<first_frame:
            last_frame=len(self.d1s[sn]['subtracted'])
        d1s0 = copy.deepcopy(self.d1s[sn]['subtracted'][first_frame])
        if last_frame>first_frame+1:
            d1s0.avg(self.d1s[sn]['subtracted'][first_frame+1:last_frame], debug=debug)
        if save_data:
            d1s0.save("%s%s_%d-%d%c.dat"%(path,sn,first_frame,last_frame-1,'s'), debug=debug)
        if plot_data:
            if fig is None:
                fig = plt.figure()            
            ax1 = fig.add_subplot(121)
            plt.semilogy(d1s0.qgrid, d1s0.data)
            plt.errorbar(d1s0.qgrid, d1s0.data, d1s0.err)
            plt.xlim(0,qmax)
            ax2 = fig.add_subplot(122)
            i0,rg = d1s0.plot_Guinier(qs=qs, ax=ax2)
            print(f"I0={i0:.2g}, Rg={rg:.2f}")
            plt.tight_layout()            
    
        
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
                d1s0.avg(d1s[1:], plot_data=plot_averaged, ax=ax, debug=debug)
            d1s0.save("%s%s_%d-%d%c.dat"%(path,sn,first_frame,last_frame-1,dkey[0]), debug=debug)
        else:
            for i in range(len(d1s)):
                d1s[i].save("%s%s_%d%c.dat"%(path,sn,i+first_frame,dkey[0]), debug=debug)                    

        
class h5sol_HT(h5xs):
    """ multiple samples, not many frames per sample
    """    
    d1b = {}   # buffer data used, could be averaged from multiple samples
    buffer_list = {}    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
                
    def update_h5(self, debug=False):
        """ raw data are updated using add_sample()
            save sample-buffer assignment
            save processed data
        """
        #fh5 = h5py.File(self.fn, "r+")
        if debug is True:
            print("updating 1d data and buffer info") 
        fh5 = self.fh5
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
            self.save_d1s(sn, debug=debug)
        #fh5.flush()                           
        
    def process(self, detectors=None, update_only=False,
                reft=-1, sc_factor=1., save_1d=False, save_merged=False, 
                filter_data=False, debug=False, N = 1):
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
            if sc_factor>0:
                self.attrs[sn]['sc_factor'] = sc_factor
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
        