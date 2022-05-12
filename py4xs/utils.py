import numpy as np
import matplotlib.colors as mc
from functools import reduce
import subprocess

def auto_clim(v, logScale, ch_thresh=10):
    if logScale:
        tt = np.log(v[v>0])
    else:
        tt = v[v>=0]
    vmax = np.max(tt)
    vmin = np.min(tt)
    
    for i in range(10):
        val,bins = np.histogram(tt, range=[vmin, vmax], bins=100)
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

def get_bin_ranges_from_grid(qgrid, prec=1e-5):
    """ convert the given qgrid, which is composed of multiple evenly spaced segments, into a sereis 
        of bin_ranges that can be used for histogramming
    """
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