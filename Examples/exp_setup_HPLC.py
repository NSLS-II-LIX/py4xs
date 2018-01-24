import os,time
from pyXS.Data2D import Data2d,Axes2dPlot,ExpParaLiX
from pyXS.slnXS import Data1d, average, merge_detectors, trans_mode, TRANS_FROM_WAXS
from pyXS.DetectorConfig import DetectorConfig
import numpy as np
import copy
import pylab as plt
from scipy.interpolate import splrep,sproot

import multiprocessing as mp

trans_mode = TRANS_FROM_WAXS

ene = 13768.86
wl = 2.*np.pi*1973/ene

es = ExpParaLiX(1043, 981) 
ew1 = ExpParaLiX(619, 487) 
ew2 = ExpParaLiX(487, 619)

es.wavelength = wl
es.bm_ctr_x = 448 #446.
es.bm_ctr_y = 766 #767.7
es.ratioDw = 23.8
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
ew1.bm_ctr_x = -213.5   
ew1.bm_ctr_y = 241   
ew1.ratioDw = 8.95
ew1.det_orient = 0.
ew1.det_tilt = -26.
ew1.det_phi = 0.
ew1.grazing_incident = False
ew1.flip = 1
ew1.incident_angle = 0.2
ew1.sample_normal = 0

ew1.calc_rot_matrix()
ew1.init_coordinates()

ew2.wavelength = wl
ew2.bm_ctr_x = 583     
ew2.bm_ctr_y = 284.5  
ew2.ratioDw = 3.19
ew2.det_orient = 0.
ew2.det_tilt = 24.
ew2.det_phi = 0
ew2.grazing_incident = False
ew2.flip = 0
ew2.incident_angle = 0.2
ew2.sample_normal = 0

ew2.calc_rot_matrix()
ew2.init_coordinates()

es.mask.read_file("XX-mask.SAXS")
ew1.mask.read_file("XX-mask.WAXS1")
ew2.mask.read_file("XX-mask.WAXS2")

qgrid = np.hstack((np.arange(0.006, 0.0499, 0.001),
                   np.arange(0.05, 0.0999, 0.002),
                   np.arange(0.1, 0.4999, 0.005),
                   np.arange(0.5, 0.9999, 0.01),
                   np.arange(1.0, 3.2, 0.03)))

det_saxs = DetectorConfig(extension="_SAXS", 
                          fix_scale = 1.0,  
                          exp_para=es, qgrid = qgrid, 
                          bad_pixels=[[   6,  76, 302, 359, 500, 751, 752, 752, 752, 753, 753, 754, 754,
                                       754, 754, 755, 755, 755, 755, 756, 756, 756, 756, 756, 757, 757,
                                       757, 758, 758, 758, 772, 773, 774, 778, 778, 779, 779, 779, 780,
                                       780, 780, 781, 781, 832, 853, 969, 979, 980], 
                                      [45, 364, 192, 513, 157, 773, 432, 445, 446, 433, 443, 433, 434,
                                       435, 441, 434, 435, 436, 440, 435, 436, 437, 438, 439, 436, 437,
                                       438, 436, 437, 438, 463, 463, 462, 458, 459, 458, 459, 460, 459,
                                       460, 461, 446, 461, 803, 920, 506, 481,  57]])
det_waxs1 = DetectorConfig(extension="_WAXS1", 
                           fix_scale = 21, #18.3, 
                           exp_para=ew1, qgrid = qgrid)
det_waxs2 = DetectorConfig(extension="_WAXS2", 
                           fix_scale = 251., 
                           exp_para=ew2, qgrid = qgrid)

detectors = [det_saxs, det_waxs1, det_waxs2]

def get_files_from_fn(fn):
    fl =[]
    fn_dir, fn_root = os.path.split(fn)
    for x in os.listdir(fn_dir):
        if x.startswith(fn_root+"_0") and (det_saxs.extension in x):
            fn1, fn2 = (fn_dir+'/'+x).split(det_saxs.extension)
            fl.append(fn1+'%s'+fn2) 
    fl.sort()
    return fl

def get_data_from_fn(fn, exp_para):
    d2s = Data2D.Data2d(fn%"_SAXS")
    d2w1 = Data2D.Data2d(fn%"_WAXS1")
    d2w2 = Data2D.Data2d(fn%"_WAXS2")
    d2s.set_exp_para(exp_para[0])
    d2w1.set_exp_para(exp_para[1])
    d2w2.set_exp_para(exp_para[2])
    return (d2s, d2w1, d2w2)

def read_Shimadzu_section(section):
    """ the chromtographic data section starts with a header
        followed by 2-column data
        the input is a collection of strings
    """
    xdata = []
    ydata = []
    for line in section:
        tt = line.split()
        if len(tt)==2:
            try:
                x=float(tt[0])
            except ValueError:
                continue
            try:
                y=float(tt[1])
            except ValueError:
                continue
            xdata.append(x)
            ydata.append(y)
    return xdata,ydata

def read_Shimadzu_datafile(fn):
    """ read the ascii data from Shimadzu Lab Solutions software
        the file appear to be split in to multiple sections, each starts with [section name], 
        and ends with a empty line
        returns the data in the sections titled 
            [LC Chromatogram(Detector A-Ch1)] and [LC Chromatogram(Detector B-Ch1)]
    """
    fd = open(fn, "r")
    lines = fd.read().split('\n')
    fd.close()
    
    sections = []
    while True:
        try:
            idx = lines.index('')
        except ValueError:
            break
        if idx>0:
            sections.append(lines[:idx])
        lines = lines[idx+1:]
    
    data = []
    for s in sections:
        if s[0][:16]=="[LC Chromatogram":
            x,y = read_Shimadzu_section(s)
            data.append([s[0],x,y])
    
    return data

def get_HPLCdata_from_fns(fn_root, save1d=False, debug='quiet'):
    """ read solution scattering data and merge inti 
    """
    fl = get_files_from_fn(fn_root)
    t1 = time.time()
    print("processing ... ",)
    data = merge_detectors(fl, detectors, plot_data=False, save_merged=save1d, qmin=0.005, qmax=2.6, debug=debug)
    t2 = time.time()
    print("done. Time elapsed = %.2f sec" % (t2-t1))
    return data


def proc(args):
    return merge_detectors(args[0], detectors, plot_data=False, save_merged=args[1], debug=args[2])


def divide_fl(fl, Nmax=8, lmin=50):
    """ split a long file list into multiple (not more than Nmax) smaller lists
        each list in no more shorter than lmin 
    """
    l = len(fl)
    N1 = int(l/lmin)+1
    if N1>Nmax:
        N1=Nmax
    
    ll = int(l/N1)
    list2 = []
    for i in range(N1):
        list2.append(fl[i*ll:(i+1)*ll])

    return list2,N1
        
import itertools

def get_HPLCdata_from_fns_mp(fn_root, save1d=False, debug='quiet'):
    """ read solution scattering data and merge inti 
    """
    fl = get_files_from_fn(fn_root)
    t1 = time.time()
    print("processing ... ",)
    
    ## start multiprocessing here
    dfl,N = divide_fl(fl)    
    pool = mp.Pool(processes=N)
    result = pool.map(proc, [(fi, save1d, debug) for fi in dfl] )

    t2 = time.time()
    print("done. Time elapsed = %.2f sec" % (t2-t1))
    return list(itertools.chain(*result))


def plot_HPLCdata(data, 
                  i_minq=0.02, i_maxq=0.05, normalize_int=False,
                  uv_file=None, flowrate=0, ymin=-1, ymax=-1, 
                  offset=0, uv_scale=1, showFWHM=False, 
                  buffer_frames=[], calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True,
                  txtfile=None, save_1d_data=False):
    """ plot the HPLC data as scattering intensity vs frame number and time or elution volume
        plot UV/RI data exported from the HPLC software, if available
        plot Rg for intensity values over a threahold value
        
        data: a collection of Data1D objects, generated from get_HPLCdata_from_fns()
        i_minq, i_maxq: regions-of-interest within with the integrated intensity is calculated
        uv_file: ascii data file exported from Shimadzu Lab Solution software
        flowrate: if this is >0, the x-axis will be shown as volume (time x flowrate) rather than time 
        ymin/ymax: manually set plot limits for the intensity axis
        uv_scale: scale the UV/RI data from the HPLC detectors
        showFWHM: calculate the width of the x-ray peak
        textfile: save the time vs. intensity data to a text file
        thresh: the intensity must be greater than thresh to perform Guinier fit and save file
        save_1d_data: if true, save the 1D data after buffer subtraction
        calc_Rg: perform Gunier fit for each frame 
        qs/qe/fix_qe: range of the Guinier fit, whether the high q end of the range should be fixed
        
        timestamp is obtained from the cbf file header
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()

    ref_water_int = -1
    
    if len(buffer_frames)>1:
        dbs = [data[i] for i in buffer_frames[1:]]
        dbuf = data[buffer_frames[0]].avg(dbs, debug='quiet')
    elif len(buffer_frames)==1:
        dbuf = data[buffer_frames[0]]
    else:
        dbuf = None
        
    ts0 = data[0].timestamp
    idx = (data[0].qgrid>i_minq) & (data[0].qgrid<i_maxq) 
    data_t = []
    data_i = []
    data_rg = []
    frame_n = 0
    for d1 in data:
        ti = (d1.timestamp-ts0).total_seconds()/60
        if flowrate>0:
            ti*=flowrate
            
        if normalize_int:
            d1.set_trans(ref_trans=data[0].trans)
        if dbuf!=None:
            dt = d1.bkg_cor(dbuf, plot_data=False, debug="quiet", sc_factor=1.0)
            ii = dt.data[idx].sum()
        else:
            ii = d1.data[idx].sum()

        if ii>thresh and calc_Rg and dbuf!=None:
            i0,rg = dt.plot_Guinier(qs, qe, fix_qe=fix_qe, no_plot=True)
            #print("frame # %d: i0=%.2g, rg=%.2f" % (frame_n,i0,rg))
        else:
            rg = 0

        if ii>thresh and save_1d_data and dbuf!=None and txtfile!=None:
            dt.save("%s-f%3d.dat" % (txtfile, frame_n))
        
        data_t.append(ti)
        data_i.append(ii)
        data_rg.append(rg)
        frame_n += 1

    if txtfile!=None:
        np.savetxt(txtfile, np.transpose([data_t, data_i]))
        
    if ymin == -1:
        ymin = np.min(data_i)
    if ymax ==-1:
        ymax = np.max(data_i)
        
    ax1.plot(data_i, 'b-')
    ax1.set_xlabel("frame #")
    ax1.set_xlim((0,len(data_i)))
    ax1.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
    ax1.set_ylabel("intensity")
    
    if uv_file:
        dc = read_Shimadzu_datafile(uv_file)
        ax2.plot(np.asarray(dc[0][1])+offset,
                 ymin+dc[0][2]/np.max(dc[0][2])*(ymax-ymin)*uv_scale, 
                 "r-", label=dc[0][0]
                )
        ax2.plot(np.asarray(dc[1][1])+offset,
                 ymin+dc[1][2]/np.max(dc[1][2])*(ymax-ymin), 
                 "g-", label=dc[1][0]
                )
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
    
    if calc_Rg and dbuf!=None:
        data_rg = np.asarray(data_rg)
        max_rg = np.max(data_rg)
        data_rg[data_rg==0] = np.nan
        ax3.plot(data_rg, 'r.', label='rg')
        ax3.set_xlim((0,len(data_rg)))
        ax3.set_ylim((0, max_rg*1.05))
        ax3.set_ylabel("Rg")
    
    leg = ax2.legend(loc='upper left', fontsize=9, frameon=False)
    leg = ax3.legend(loc='center left', fontsize=9, frameon=False)
    plt.show()

    
def average_HPLCdata(data, start, end, **kargs):
    if start<end:
        d1 = data[start].avg(data[start+1:end], kargs)
    else:
        d1 = data[start]
    return d1

    
def batch_subtract_HPLCdata(data, b_start, b_end, sc_factor=0.995, debug=False):
    if b_start<b_end:
        dbuf = data[b_start].avg(data[b_start+1:b_end])
    else:
        dbuf = data[b_start]
    for i in range(len(data)):
        #if i>=b_start and i<=b_end:
        #    continue
        dt = data[i].bkg_cor(dbuf, sc_factor=sc_factor)
        if debug:
            print("saving data: %s" % (data[i].label+".dat"))
        dt.save(data[i].label+".dat")
    
    
def subtract_HPLCdata(data, s_start, s_end, b_start, b_end, sc_factor=0.995, qs=0.01, qe=0.04, fix_qe=True):
    """ data: the same list of Data1D objects used as input for plot_HPLCdata()
        s_start, s_end, b_start, b_end: 
            starting/ending frame numbers to be userd as sample/buffer data 
        qs, qe: range for Gunier fit
        returns a Data1D object
    """
    ds = average_HPLCdata(data, s_start, s_end)         
    db = average_HPLCdata(data, b_start, b_end)         
        
    d1 = ds.bkg_cor(db, plot_data=True, debug=True, sc_factor=sc_factor)
    plt.figure()
    i0,rg = d1.plot_Guinier(qs, qe, fix_qe=fix_qe)
    print("I0=%g, Rg=%.1f" % (i0,rg))
    
    return d1


