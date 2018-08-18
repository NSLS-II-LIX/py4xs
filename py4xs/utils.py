import numpy as np
import matplotlib.colors as mc
from functools import reduce

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

