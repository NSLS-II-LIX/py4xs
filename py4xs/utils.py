import numpy as np
import matplotlib.colors as mc
from functools import reduce
import subprocess

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


def smooth(x,half_window_len=11,window='hanning'):
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

def run(cmd, path="", ignoreErrors=True, returnError=False):
    """ cmd should be a list, e.g. ["ls", "-lh"]
        path is for the cmd, not the same as cwd
    """
    cmd[0] = path+cmd[0]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if len(err)>0 and not ignoreErrors:
        print(err.decode())
        raise Exception(err.decode())
    if returnError:
        return out.decode(),err.decode()
    else:
        return out.decode()