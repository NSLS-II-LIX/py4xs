import numpy as np
import pylab as plt
import matplotlib.colors as mc
from functools import reduce
import subprocess,re,os

import skimage.morphology as morph
import skimage.filters as ft

def auto_clim(v, logScale, ch_thresh=10):
    if logScale:
        tt = np.log(v[v>0])
    else:
        tt = v[v>=0]
    vmax = np.max(tt)
    vmin = np.min(tt)
    
    Nd = len(tt)
    Nbins=100
    if ch_thresh>np.sqrt(Nd)/2:
        ch_thresh = np.sqrt(Nd)/2
    if Nd/Nbins<ch_thresh:
        Nbins = int(Nd/ch_thresh)
    
    for i in range(10):
        val,bins = np.histogram(tt, bins=Nbins)
        #val,bins = np.histogram(tt, range=[vmin, vmax], bins=Nbins)
        vidx = np.where(val>ch_thresh)[0]
        imin = vidx[0]
        vmin = bins[imin]
        imax = vidx[-1]
        vmax = bins[imax+1]
        if imax-imin>ch_thresh:
            break
    clim = (vmin, vmax)
    if logScale:
        clim = np.exp(clim)    
    
    return clim

def get_bin_ranges_from_grid(qgrid, tol=0.01):
    """ break down non-uniform qgrid into multiple evenly spaced ranges 
        [[[min1, max1], N1], [[min2, max2], N2], ...]
        there will be a gap between neighonring ranges
    """
    bin_range = []
    while True:
        qmin = qgrid[0]
        dq = qgrid[1]-qgrid[0]
        pq = qmin+dq*np.arange(len(qgrid))
        N = len(qgrid[np.fabs(qgrid-pq)<dq*tol])
        bin_range.append([[qmin-0.5*dq, qmin+(N-0.5)*dq], N])
        if N==len(qgrid):
            break
        qgrid = qgrid[N:]
        
    return bin_range

""" this is incorrect
def get_bin_ranges_from_grid(qgrid, prec=1e-5):
    ''' convert the given qgrid, which is composed of multiple evenly spaced segments, into a sereis 
        of bin_ranges that can be used for histogramming
    '''
    sc = int(1./prec)
    df = np.floor(np.diff(qgrid)*sc+0.5)
    df = np.append(df, df[-1])
    bin_ranges = []
    for v in np.unique(df):
        idx = (np.fabs(df-v)<0.1)
        llmt = np.floor(qgrid[idx][0]*sc-v/2+0.5)*prec 
        hlmt = np.floor(qgrid[idx][-1]*sc+v/2+0.5)*prec 
        bin_ranges.append([[llmt, hlmt],len(df[idx])])
    
    return bin_ranges
"""

def get_grid_from_bin_ranges(bin_ranges):
    """ construct a qgrid from the given bin_ranges, which should not overlap and have the format of 
        [[[min_v, max_v], N], ...]
    """
    ql = [np.linspace(rn[0], rn[1], n+1) for rn,n in bin_ranges]
    qgrid = np.hstack([(r[1:]+r[:-1])/2 for r in ql])
    
    return qgrid

def calc_avg(dat:list, err:list, method="simple"):
    """ both dat and err should be lists of numpy arrays, corresponding to data and error bar
        calculate weighted average the data and the corresponding error 
        use 1/err^2 as the weight
    """
    if not method in ["simple", "err_weighted"]:
        raise Exception("method must be either 'simple' or 'err_weighted'.")
    
    da = np.zeros_like(dat[0])
    ea = np.zeros_like(dat[0])
    wt = np.zeros_like(dat[0])
    
    if method=="err_weighted":
        if len(err)==0:
            raise Exception("error bars required for err_weighted average.")
        for i in range(len(dat)):
            idx = (err[i]>0)
            wt[~idx] = 0 
            wt[idx] = 1./err[i][idx]**2
            da[idx] += wt[idx]*dat[i][idx]
            ea[idx] += wt[idx]
        idx = (ea>0)
        da[idx] /= ea[idx]
        ea[idx] = 1./np.sqrt(ea[idx])
    else:
        for i in range(len(dat)):
            idx = ~np.isnan(dat[i])
            da[idx] += dat[i][idx]
            if len(err)>0:
                ea[idx] += err[i][idx]
            wt[idx] += 1.
        idx = (wt>0)
        da[idx] /= wt[idx]
        if len(err)>0:
            ea[idx] = ea[idx]/wt[idx]/np.sqrt(wt[idx])        
        else: 
            ea = None

    return da,ea


def max_len(d1a, d1b, return_all=False):
    """ perform a Cormap-like pairwise comparison between 2 arrays
        if return_all is True, return a sequence of True/False base on the compasion of 
            the 2 arrays, as well as the sequence of "patch" (consecutive True/False) sizes
        otherwise return only the largest patch size
    """ 
    idx = ~(np.isnan(d1a) |  np.isnan(d1b))
    seq0 = d1a[idx]>d1b[idx]
    seq1 = ~seq0
    dif0 = np.diff(np.where(np.concatenate(([seq0[0]], seq0[:-1] != seq0[1:], [True])))[0])[::2]
    dif1 = np.diff(np.where(np.concatenate(([seq1[0]], seq1[:-1] != seq1[1:], [True])))[0])[::2]
    dif = np.hstack((dif0, dif1))
    
    if return_all:
        return seq1,dif

    return np.max(dif)


def Schilling_p_value(n, C):
    """ see Franke et.al. 2015, or Schilling 1990
        the probability of having longest patch size of C or larger with data size of n is
            1 - A(n,C)/2^n
        A(n,C) is the number of n-length sequences with the longest run not exceeding C 
        A(n,C) is given by 2^n if n<=C
        Otherwise it is calculated interatively: 
            sum{j=0,C | A_(n-1-j, C)}            
    """
    
    # first construct An(C)
    AnC = np.zeros(n+1)
    for i in range(C+1):
        AnC[i] = np.power(2.0, i)
    for i in range(C+1, n+1):
            AnC[i]= AnC[:i][-C-1:].sum()
    
    return 1.0-AnC[-1]/np.power(2.0, n)


def strip_name(s):
    strs = ["_SAXS","_WAXS1","_WAXS2",".cbf",".tif"]
    for ts in strs:
        if ts in s:
            ss = s.split(ts)
            s = "".join(ss)
    return s


def common_name(s1, s2):
    s1 = strip_name(s1)
    s2 = strip_name(s2)
    l = len(s1)
    if len(s2) < l:
        l = len(s2)

    s = ""
    for i in range(l):
        if s1[i] == s2[i]:
            s += s1[i]
        else:
            break
    if len(s) < 1:
        s = s1.copy()
    return s.rstrip("-_ ")


def reduced_cmap(cmap, step):
    return np.array(cmap(step)[0:3])


def cmap_map(function, cmap):
    """ from scipy cookbook
    Applies function (which should operate on vectors of shape 3: [r, g, b], on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = [x[0] for x in cdict[key]]
    step_list = reduce(lambda x, y: x + y, list(step_dict.values()))
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    # reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    # old_lut = np.array(list(map(reduced_cmap, step_list)))
    # new_lut = np.array(list(map(function, old_lut)))

    old_lut = np.array([reduced_cmap(cmap, s) for s in step_list])
    new_lut = np.array([function(s) for s in step_list])

    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_lut[j, i]
            elif new_lut[j, i] != old_lut[j, i]:
                this_cdict[step] = new_lut[j, i]
        colorvector = [x + (x[1],) for x in list(this_cdict.items())]
        colorvector.sort()
        cdict[key] = colorvector

    return mc.LinearSegmentedColormap('colormap', cdict, 1024)


def smooth(x, half_window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    revised from numpy cookbook, https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    window_len = 2*half_window_len+1
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    #x = np.hstack((x, np.ones(window_len)*x[-1]))

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[half_window_len:-half_window_len]

def run(cmd, path="", ignoreErrors=True, returnError=False, debug=False):
    """ cmd should be a list, e.g. ["ls", "-lh"]
        path is for the cmd, not the same as cwd
    """
    cmd[0] = path+cmd[0]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if debug:
        print(out.decode(), err.decode())
    if len(err)>0 and not ignoreErrors:
        print(err.decode())
        raise Exception(err.decode())
    if returnError:
        return out.decode(),err.decode()
    else:
        return out.decode()

def check_bm_center(dt, sn=None, det_ext='_SAXS', frn=0, 
                    qs=0.005, qe=0.05, qn=100, Ns=9, **kwargs):
    """ dt is an instance of h5xs
        
        this function compares the beam intensity on both sides of the beam center,
        and advise if the beam center as defined in the detector configuration is incorrect
        dividing data into 4*Ns slices, show data in horizontal and vertical cuts
    """
    i = 0
    d2 = dt.get_d2(sn=sn, det_ext=det_ext, frn=frn, **kwargs)
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
    
def gen_p900k_mask(img, radius=3, thresh=1.5):
    modules = [[0,195,0,487], [0,195,493,981],
               [212,407,0,487], [212,407,493,981],
               [424,619,0,487], 
               [636,831,0,487], [636,831,493,981],
               [848,1043,0,487], [848,1043,493,981]]

    msk1 = (img<0)
    msk2 = (img<1)
    for mm in modules:
        img0 = img[mm[0]:mm[1], mm[2]:mm[3]]
        img_s1 = ft.gaussian(img0, 1, preserve_range=True)
        img_s2 = ft.gaussian(img0, radius, preserve_range=True)
        img_d = img_s2-img_s1
        msk = (img_d>np.sqrt(img_s1)*thresh)
        msk1[mm[0]:mm[1], mm[2]:mm[3]] = morph.dilation(msk, morph.disk(radius))
        msk = (img_d<-np.sqrt(img_s1)*thresh)
        msk2[mm[0]:mm[1], mm[2]:mm[3]] = morph.dilation(msk, morph.disk(radius))  # used to be |= instead of =   
    
    msk1[424:619,493:981] = True
    return msk1|msk2

def auto_fix_WAXS_mask(dt, sn=None, frn=None, radius=3, replace=True, **kwargs):
    """ this is meant to deal with the mica peaks that show up in WAXS patterns
        currently can only deal with the LiX pilatus900K
    """
    for i in range(len(dt.detectors)):
        ext = dt.detectors[i].extension
        dn = det_model[ext]
        if not re.match('pilatus[\\s\\w]+900k', dn, re.IGNORECASE):
            continue
        
        img = dt.get_d2(sn=sn, det_ext=ext, frn=frn, **kwargs).data.d
        msk0 = gen_p900k_mask(img, radius=radius)
            
        if replace:
            dt.detectors[i].exp_para.mask.map = msk0
        else:
            dt.detectors[i].exp_para.mask.map |= msk0
        dt.save_detectors()

def calib_detector(det, dstd, sn, wl, det_type, pxsize, 
                   bkg, temp_file_location, use_existing_poni=False):
    print(f"processing detector {det.extension} ...")    
    ep = det.exp_para
    ep.wavelength = wl
    uname = os.getenv("USER")
    data_file = f"{temp_file_location}/{uname}{det.extension}.cbf"

    if not use_existing_poni:
        with h5py.File(dstd.fn, "r") as fh5:
            dn = dstd.det_name[det.extension]
            strn = find_field(fh5, dn, sn)
            img = fh5[f"{sn}/{strn}/data/{dn}"][0]
    
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
            #fabio.cbfimage.CbfImage(data=img*(~dmask)).write(data_file)
            fabio.cbfimage.CbfImage(data=img).write(data_file)

    if use_existing_poni:
        fp = use_existing_poni
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
    ep.Dd = float(lines['Distance'])/pxsize
    D = ep.Dd/np.dot(np.dot(ep.rot_matrix, np.asarray([0, 0, 1.])), np.asarray([0, 0, 1.])) 
    ep.ratioDw = D/(ep.ImageWidth)
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
    
def generate_mask_from_std(det, dstd, std_samples=None, template_map=None, thresh=1000, mica_window=True):
    """ some LiX-specific assumptions are made
        all pixels with counts higher than the specified thresh value are discarded
    """

    if std_samples is None:
        std_samples = {}
        std_samples["empty"] = []
        for sn in dstd.samples:
            if re.search('AgBH', sn, re.IGNORECASE):
                std_samples['AgBH'] = sn
            elif re.search('empty', sn, re.IGNORECASE):
                std_samples['empty'].append(sn)
            elif re.search('dark', sn, re.IGNORECASE):
                std_samples['dark'] = sn
            elif re.search('carbon', sn, re.IGNORECASE):
                std_samples['carbon'] = sn
        
        if not set(["dark", "carbon"]).issubset(std_samples.keys()) or len(std_samples["empty"])==0:
            raise Exception(f"Not all required data are present for mask generation ...: {dstd_samples.keys()}")
    
    d2dark = dstd.get_d2(sn=std_samples['dark'], det_ext=det.extension, frn='sum').data.d
    d2carbon = dstd.get_d2(sn=std_samples['carbon'], det_ext=det.extension, frn='sum').data.d
    d2empty = np.average([dstd.get_d2(sn=sn, det_ext=det.extension, frn='sum').data.d for sn in std_samples['empty']], axis=0)
    
    hot_pix = (d2dark>10)
    if "SAXS" in det.extension:
        cstd = np.std(d2carbon[d2carbon<thresh])
        cc,bb = np.histogram(d2carbon.flatten(), bins=50, range=[cstd*0.05, cstd*2])
        bb0 = (bb[:-1]+bb[1:])/2
        carbon_thresh = np.average(bb0[cc<2*np.min(cc[cc>0])])

        bs = (d2carbon<carbon_thresh) & (d2carbon>0)
        bs = (morph.dilation(bs, morph.disk(2))>0)
        extra = bs
    else: # WAXS
        d2empty1 = ft.gaussian(d2empty, 2)
        # this may need another look; affected by mica peaks
        cstd = np.std(d2empty1[d2empty1<thresh])
        cc,bb = np.histogram(d2empty1.flatten(), bins=50, range=[cstd*0.1, cstd*10])
        bb0 = (bb[:-1]+bb[1:])/2
        cc0 = cc[-1]/2 
        empty_thresh = np.average(bb0[cc>0.01*np.max(cc)])*3   # 1% of max should eliminate most air scattering
        
        mica_pks = (d2empty1>empty_thresh)
        if mica_window:
            extra = mica_pks
        else:
            extra = np.zeros_like(mica_pks)

        # based on observation, 
        # the bin with the lowest intensity also has the msot counts
        # anything below should be dead pixels
        cc1,bb1 = np.histogram(d2carbon[hot_pix|extra==0].flatten(), bins=100, 
                             range=[0, np.std(d2carbon)*0.1])
        carbon_thresh = bb1[np.argmax(cc1[1:])]/5   # this scaling factor is sensitive         

    dead_pix = (d2carbon<carbon_thresh)
    return hot_pix|dead_pix|extra
    
