### copied from 2018C2 version of h5sol.py

import pylab as plt
import h5py
import numpy as np
import time, datetime
import json

import multiprocessing as mp

from py4xs.slnxs import Data1d, average
from py4xs.utils import common_name
from py4xs.slnxs import filter_by_similarity

det_name = {"_SAXS": "pil1M_image",
            "_WAXS1": "pilW1_image",
            "_WAXS2": "pilW2_image",
           }

def lsh5(hd, prefix='', top_only=False, silent=False):
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

def pack_d1(data):
    if isinstance(data, Data1d):
        return np.asarray([data.data,data.err])
    elif isinstance(data, list):
        return np.asarray([pack_d1(d) for d in data])
    
def unpack_d1(data, qgrid, label):
    if len(data.shape)>2:
        return [unpack_d1(d, qgrid, label+("f%05d" % i)) for i,d in enumerate(data)]
    else:
        ret = Data1d()
        ret.qgrid = qgrid
        ret.data = data[0]
        ret.err = data[1]
        ret.label = label
        ret.set_trans()   
        return ret

def merge_d1s(d1s, detectors, reft=-1, save_merged=False, debug=False):
    s0 = Data1d()
    s0.qgrid = d1s[0].qgrid
    d_tot = np.zeros(detectors[0].qgrid.shape)
    d_max = np.zeros(detectors[0].qgrid.shape)
    d_min = np.zeros(detectors[0].qgrid.shape)+1.e32
    e_tot = np.zeros(detectors[0].qgrid.shape)
    c_tot = np.zeros(detectors[0].qgrid.shape)
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
    s0.overlaps.append({'q_overlap': detectors[0].qgrid[idx],
                        'raw_data1': d_max[idx],
                        'raw_data2': d_min[idx]})
    s0.data[idx] /= c_tot[idx]
    s0.err[idx] /= np.sqrt(c_tot[idx])
    s0.set_trans(ref_trans=reft, debug=debug)
    s0.label = label
    s0.comments = comments # .replace("# ", "## ")
    if save_merged:
        s0.save(dt.label+".dd", debug=debug)
        
    return s0


def proc_sample(queue, images, sn, nframes, detectors, reft, save_1d, save_merged, debug,
               starting_frame_no=0):
    ret = {'merged': []}
    sc = {}
    
    for det in detectors:
        ret[det.extension] = []
        if det.fix_scale is not None:
            sc[det.extension] = 1./det.fix_scale

    if debug:
        print("processing started: sample = %s, starting frame = #%d" % (sn, starting_frame_no))
    for i in range(nframes):
        for det in detectors:
            dt = Data1d()
            label = "%s_f%05d%s" % (sn, i+starting_frame_no, det.extension)
            dt.load_from_2D(images[det.extension][i+starting_frame_no], 
                            det.exp_para, det.qgrid, det.pre_process, det.mask,
                            save_ave=False, debug=debug, label=label)
            dt.scale(sc[det.extension])
            ret[det.extension].append(dt)
    
        ret['merged'].append(merge_d1s([ret[det.extension][i] for det in detectors],
                                       detectors, reft, save_merged, debug))

    queue.put([sn,starting_frame_no,ret])
    if debug:
        print("processing completed: ", sn, starting_frame_no)


class h5sol():
    d1s = {}
    detectors = None
    samples = []
    attrs = {}
    
    def __init__(self, fn, detectors):
        self.detectors = detectors
        self.fn = fn
        
    # for each sample
    #      attribute "selected": which raw data are included in average
    #      attribute "sc_factor": scaling factor used for buffer subtraction
    # raw data: from each detector, and 'merged'
    # averaged data: averaged from multiple frames
    # corrected data: subtracted for buffer scattering
    # buffer data will be recreated based on buffer assignment 
    def load_d1s(self, fh5, sn):
        if "processed" not in lsh5(fh5[sn], top_only=True, silent=True): return
        
        if sn not in list(self.attrs.keys()):
            self.d1s[sn] = {}
            self.attrs[sn] = {}
        grp = fh5[sn+'/processed']
        for k in list(grp.attrs.keys()):
            self.attrs[sn][k] = grp.attrs[k]   
        qgrid = grp['qgrid'].value
        #self.d1s[sn]['buf average'] = grp['buf average'].value
        for k in lsh5(grp, top_only=True, silent=True):
            if k=='qgrid':
                continue
            elif k=='buf average':
                self.d1s[sn][k] = grp[k].value
            else:
                self.d1s[sn][k] = unpack_d1(grp[k], qgrid, sn+k)           
        
    # save the qgrid just once, since all data1d share the same qgrid
    # processed data go under the group sample_name/processed
    def save_d1s(self, fh5, sn, debug=False):
        # assume that the shape of the data/qgrid is unchanged
        if "processed" not in list(lsh5(fh5[sn], top_only=True, silent=True)):
            grp = fh5[sn].create_group("processed")
        else:
            grp = fh5[sn+'/processed']
        
        for k in list(self.attrs[sn].keys()):
            grp.attrs[k] = self.attrs[sn][k]
            if debug:
                print("writting attribute to %s: %s" % (sn, k))

        ds_names = lsh5(grp, top_only=True, silent=True)
        if 'qgrid' not in ds_names:
            grp.create_dataset('qgrid', data=self.d1s[sn]['merged'][0].qgrid)
        else:
            grp['qgrid'][...] = self.d1s[sn]['merged'][0].qgrid
        for k in list(self.d1s[sn].keys()):
            data = pack_d1(self.d1s[sn][k])
            if k not in ds_names:
                grp.create_dataset(k, data=data)
            else:
                grp[k][...] = data   
            if debug:
                print("writting attribute to %s: %s" % (sn, k))
        
    def make_1d(self, reft=-1, save_1d=False, save_merged=False, debug=False):
        """ produce merged 1D data from 2D images
        """
        if debug:
            print("start processing: make_1d()")
            t1 = time.time()
        
        fh5 = h5py.File(self.fn, "r+")
        
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        for sn in self.samples:
            if debug:
                print("reading data for ", sn)
            self.d1s[sn] = []
            images = {}
            for det in self.detectors:
                data = fh5["%s/primary/data/%s" % (sn, det_name[det.extension])].value 
                if len(data.shape)==4:
                    # quirk of suitcase
                    data = data[0]
                images[det.extension] = data  
            qsize = len(images[self.detectors[0].extension])
            for i in range(qsize):
                ret = {}
                for det in self.detectors:
                    dt = Data1d()
                    label = "%s_f%05d_%s" % (sn, i, det.extension)
                    dt.load_from_2D(images[det.extension][i], det.exp_para, 
                                    det.qgrid, det.pre_process, det.mask,
                                    save_ave=False, debug=debug, label=label)
                    ret[det.extension] = dt
                dt = Data1d()
                dt = merge_d1s(list(ret.values()), self.detectors, reft, save_merged, debug)
                self.d1s[sn].append(dt) 
                
        fh5.close()
        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))


class h5sol_HPLC(h5sol):
    """ single sample, many frames
        frames can be added gradually
    """ 
    dbuf = None
    updating = False   # this is set to True when add_data() is active
        
    def add_data(self, ):
        """ watch the given path
            update self.d1s when new files are found
            save to HDF when the scan is done
        """
        pass
    
    def process_sample_name(self, sn, debug=False):
        fh5 = h5py.File(self.fn, "r+")
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        if sn==None:
            sn = self.samples[0]
        elif sn not in self.samples:
            raise Exception(sn, "not in the sample list.")
        
        return fh5,sn 
        
    def load_data_mp(self, update_only=False, sn=None,
                   reft=-1, save_1d=False, save_merged=False, debug=False, N=8):
        """ assume single sample, parallel-process by grouping frames into chunks
            if update_only is true, only create 1d data for new frames (empty self.d1s)
        """
        if debug:
            print("start processing: load_1d()")
            t1 = time.time()
        
        fh5,sn = self.process_sample_name(sn)
        # this should depend on update_only, for now reload everything ??
        # but load_d1s() fills the class as well
        #self.load_d1s(fh5, sn)   # only one sample for HPLC
        if sn not in list(self.attrs.keys()):
            self.d1s[sn] = {}
            self.attrs[sn] = {}
            
        processes = []
        queue_list = []

        # only need to do this for new frames
        images = {}
        for det in self.detectors:
            ti = fh5["%s/primary/data/%s" % (sn, det_name[det.extension])].value
            if len(ti.shape)==4:
                ti = ti[0]      # quirk of suitcase
            images[det.extension] = ti
            
        n_total_frames = len(images[self.detectors[0].extension])
        c_size = int(n_total_frames/N)
        for i in range(N):
            if i==N-1:
                nframes = n_total_frames - c_size*(N-1)
            else:
                nframes = c_size
            que = mp.Queue()
            queue_list.append(que)
            th = mp.Process(target=proc_sample, 
                            args=(que, images, sn, nframes,
                                  self.detectors, reft, save_1d, save_merged, debug, i*c_size) )
            th.start()
            processes.append(th)
                
        result = {}
        for que in queue_list:
            [sn, fr1, data] = que.get() 
            result[fr1] = data

        for th in processes:
            th.join()
        if debug:
            print("all processes completed.")
        
        dns = [det.extension for det in self.detectors]+["merged"]
        frns = list(result.keys())
        frns.sort()
        for dn in dns:
            self.d1s[sn][dn] = []
            for frn in frns:
                self.d1s[sn][dn] += result[frn][dn]
        
        self.save_d1s(fh5, sn, debug=debug)
        fh5.close()

        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))


    def subtract_buffer(self, buffer_frame_range, sample_frame_range=None,
                        sn=None, update_only=False, sc_factor=1., normalize_int=True, debug=False):
        """ buffer_frame_range should be a list of frame numbers, could be range(frame_s, frame_e)
            if sample_frame_range is None: subtract all dataset; otherwise subtract and test-plot
            update_only ??
        """

        fh5,sn = self.process_sample_name(sn, debug=debug)
        
        if debug:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        listb  = [self.d1s[sn]['merged'][i] for i in buffer_frame_range]
        listbfn = [i for i in buffer_frame_range]
        if len(listb)>1:
            d1b = listb[0].avg(listb[1:], debug=debug)
        else:
            d1b = listb[0]           
            
        if sample_frame_range==None:
            # perform subtraction on all data and save listbfn, d1b
            self.attrs[sn]['buffer frames'] = listbfn
            self.attrs[sn]['sc_factor'] = sc_factor
            self.d1s[sn]['buf average'] = d1b
            if 'subtracted' in self.d1s[sn].keys():
                del self.d1s[sn]['subtracted']
            self.d1s[sn]['subtracted'] = []
            for d1 in self.d1s[sn]['merged']:
                if normalize_int:
                    d1.set_trans(ref_trans=self.d1s[sn]['merged'][0].trans)
                d1t = d1.bkg_cor(d1b, plot_data=False, debug="quiet", sc_factor=sc_factor)
                self.d1s[sn]['subtracted'].append(d1t) 
            self.save_d1s(fh5, sn, debug=debug)
        else:
            lists  = [self.d1s[sn]['merged'][i] for i in sample_frame_range]
            if len(listb)>1:
                d1s = lists[0].avg(lists[1:], debug=debug)
            else:
                d1s = lists[0]
            sample_sub = d1s.bkg_cor(d1b, plot_data=True, debug=debug, sc_factor=sc_factor)
        fh5.close()
        
        #if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
        #if sn not in list(self.buffer_list.keys()): continue

        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    def plot_data(self, sn=None, i_minq=0.02, i_maxq=0.05, flowrate=0, plot_merged=False,
                  ymin=-1, ymax=-1, offset=0, uv_scale=1, showFWHM=False, 
                  calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True,
                  plot2d=True, logScale=True, clim=[1.e-3, 10.],
                  debug=False):
        """ plot "merged" if no "subtracted" present
            
        """
        
        if plot2d:
            plt.figure(figsize=(8, 10))
            plt.subplot(211)
        else:
            plt.figure(figsize=(8, 6))

        fh5,sn = self.process_sample_name(sn, debug=debug)

        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax3 = ax1.twinx()
        
        if 'subtracted' in self.d1s[sn].keys() and plot_merged==False:
            dkey = 'subtracted'
        elif 'merged' in self.d1s[sn].keys():
            dkey = 'merged'
        else:
            raise Exception("processed data not present.")
            
        data = self.d1s[sn][dkey]
        qgrid = data[0].qgrid
        ts = fh5[sn+'/primary/time'].value
        idx = (qgrid>i_minq) & (qgrid<i_maxq) 
        data_t = []
        data_i = []
        data_rg = []
        ds = []
        frame_n = 0
        for i in range(len(data)):
            ti = (ts[i]-ts[0])/60
            if flowrate>0:
                ti*=flowrate

            #if normalize_int:
            #    data[i].set_trans(ref_trans=data[0].trans)
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
            ext = [0, len(data), qgrid[-1], qgrid[0]]
            asp = len(data)/qgrid[-1]/2
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

        plt.show()

        
class h5sol_HT(h5sol):
    """ multiple samples, not many frames per sample
    """    
    d1b = {}   # buffer data used, could be averaged from multiple samples
    buffer_list = {}    
    
    def list_samples(self):
        print(self.samples)
    
    def add_sample(self, db, uid):
        """ add another group to the HDF5 file
            only works at the beamline
        """
        header = db[uid]
        
    def assign_buffer(self, buf_list):
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
    
    def update_h5(self, debug=False):
        """ raw data are updated using add_sample()
            save sample-buffer assignment
            save processed data
        """
        fh5 = h5py.File(self.fn, "r+")
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
            self.save_d1s(fh5, sn, debug=debug)
        fh5.close()                           
        
    def process(self, detectors=None, update_only=False,
                reft=-1, save_1d=False, save_merged=False, 
                filter_data=False, debug=False):
        if detectors is not None:
            self.detectors = detectors
        self.load_data_mp(update_only=update_only, reft=reft, 
                          save_1d=save_1d, save_merged=save_merged, debug=debug)
        self.average_samples(update_only=update_only, filter_data=filter_data, debug=debug)
        self.subtract_buffer(update_only=update_only, debug=debug)
        
    def load_data_mp(self, update_only=False,
                   reft=-1, save_1d=False, save_merged=False, debug=False, N=8):
        """ assume multiple samples, parallel-process by sample
            if update_only is true, only create 1d data for new frames (empty self.d1s)
        """
        if debug:
            print("start processing: load_data_mp()")
            t1 = time.time()
        
        fh5 = h5py.File(self.fn, "r+")
        self.samples = lsh5(fh5, top_only=True, silent=(not debug))
        
        processes = []
        queue_list = []
        for sn in self.samples:
            if 'buffer' in list(fh5[sn].attrs):
                self.buffer_list[sn] = fh5[sn].attrs['buffer'].split('  ')
            self.load_d1s(fh5, sn)   # load processed data saved in the file
            if update_only and sn in list(self.d1s.keys()):
                continue
                
            images = {}
            for det in detectors:
                ti = fh5["%s/primary/data/%s" % (sn, det_name[det.extension])].value
                if len(ti.shape)==4:
                    ti = ti[0]      # quirk of suitcase
                images[det.extension] = ti
            
            nframes = len(images[detectors[0].extension])
            que = mp.Queue()
            queue_list.append(que)
            th = mp.Process(target=proc_sample,
                            args=(que, images, sn, nframes, 
                                  detectors, reft, save_1d, save_merged, debug) )
            th.start()
            processes.append(th)

        for th in processes:
            th.join()
        for que in queue_list:
            [sn, fr1, data] = que.get() 
            print("data received: ", sn)
            self.d1s[sn] = data
            self.save_d1s(fh5, sn, debug=debug)
            
        fh5.close()        
        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
      
            
    def average_samples(self, samples=None, update_only=False, filter_data=False, debug=False):
        if debug:
            print("start processing: average_samples()")
            t1 = time.time()

        if samples is None:
            samples = self.samples
        elif isinstance(samples, str):
            samples = [samples]
        for sn in samples:
            if update_only and 'merged' in list(self.d1s[sn].keys()): continue
                
            d1keep,d1disc = filter_by_similarity(self.d1s[sn]['merged'])
            self.attrs[sn]['selected'] = []
            for d1 in self.d1s[sn]['merged']:
                self.attrs[sn]['selected'].append(d1 in d1keep)
                
            if len(d1keep)>1:
                self.d1s[sn]['averaged'] = d1keep[0].avg(d1keep[1:], debug=debug)
            else:
                self.d1s[sn]['averaged'] = d1keep[0]

        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    def subtract_buffer(self, samples=None, update_only=False, sc_factor=1., debug=False):

        if samples is None:
            samples = list(self.buffer_list.keys())
        elif isinstance(samples, str):
            samples = [samples]

        if debug:
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
            self.d1s[sn]['subtracted'] = self.d1s[sn]['averaged'].bkg_cor(self.d1b[sn], 
                                                                          sc_factor =sc_factor)
            self.attrs[sn]['sc_factor'] = sc_factor

        if debug:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
                
    def plot_sample(self, sn, ax=None, offset=1.5, show_subtracted=False):
        """ if background-subtracted, show both sample and buffer 
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()
        if show_subtracted:
            if 'subtracted' not in list(self.d1s[sn].keys()):
                raise Exception("sample not found/bkg-subtracted: ", sn)
            self.d1s[sn]['subtracted'].plot(ax=ax)
            self.d1s[sn]['averaged'].plot(ax=ax)
            self.d1b[sn].plot(ax=ax)
        else:
            sc = 1
            for i,d1 in enumerate(self.d1s[sn]['merged']):
                if self.attrs[sn]['selected'][i]:
                    d1.plot(ax=ax, scale=sc)
                    plt.plot(self.d1s[sn]['averaged'].qgrid, 
                             self.d1s[sn]['averaged'].data*sc, 
                             color="gray", lw=2, ls="--")
                else:
                    plt.plot(self.d1s[sn]['merged'][i].qgrid, self.d1s[sn]['merged'][i].data*sc, 
                             color="gray", lw=2, ls=":")
                sc *= offset
                
