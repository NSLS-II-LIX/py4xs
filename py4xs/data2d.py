import numpy as np
import fabio
import datetime,os,copy
from py4xs.mask import Mask
from py4xs.local import ExpPara
from py4xs.utils import calc_avg,auto_clim
import pylab as plt
import matplotlib as mpl
from enum import Enum 
from PIL import Image
#import fast_histogram as fh

class DataType(Enum):
    det = 1
    qrqz = 2
    qphi = 3
    q = 4
    xyq = 5      # Cartesian version of qphi
    

def get_bin_edges(grid):
    """ find the bin edges that correspond to the desired data grid
        the average of neghboring bin edges reproduces the position in the grid 
    """
    v0 = grid[0]-(grid[1]-grid[0])/2
    edges = [v0]
    for i in range(len(grid)):
        v1 = grid[i]*2 - v0
        edges.append(v1)
        v0 = v1
    return np.asarray(edges)

def get_bins_range(qgrid, tol=0.01):
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

def round_by_stepsize(v, ss):
    prec = -int(np.floor(np.log10(np.fabs(ss))))
    if prec<1:
        prec = 1
    return f"{v:.{prec}f}".rstrip('0')

def grid_labels(grid, N=3, step_tol=0.2):
    dd = np.diff(grid)
    tt = []
    td = []
    glist = []
    for i in range(len(dd)):
        if len(tt)>0: 
            if dd[i]-tt[-1]>step_tol*dd[i]:
                glist.append([np.mean(tt), td])
                tt = []  
                td = []
        tt.append(dd[i])
        td.append(grid[i])
    td.append(grid[i+1])
    glist.append([np.mean(tt), td])
    
    if len(glist)==1: # single step size
        ss = glist[0][0]
        gpindex = np.linspace(1, len(grid), N, dtype=int)-1
        gpvalues = [grid[k] for k in gpindex]
        gplabels = [round_by_stepsize(gp,ss) for gp in gpvalues]
    else:
        gpindex = []
        gpvalues = []
        gplabels = []
        i = 0
        glist
        for [ss, gp] in glist: 
            gpindex.append(i)
            gpvalues.append(gp[0])
            gplabels.append(round_by_stepsize(gp[0],ss))
            i += len(gp)
        gpindex.append(i)
        gpvalues.append(gp[-1])
        gplabels.append(round_by_stepsize(gp[-1],ss))
    
    # on a 2D plot, the pixel size is finite, the extent of the axis goes from -0.5 to n-0.5
    # it appears that the only way to align the ticks on a twin axis is to have a blank at the end
    n = len(grid)
    gpindex = np.append(0.5+np.array(gpindex), n)
    gplabels.append('')
    
    return gpindex,gpvalues,gplabels

def histogram2d(x, y, range, bins, weights):
    return np.histogram2d(x, y, range=range, bins=bins, weights=weights)
    #return fh.histogram2d(x, y, range=range, bins=bins, weights=weights)

class MatrixWithCoords:
    # 2D data with coordinates
    d = None
    xc = None
    xc_label = None
    xc_prec = 3
    yc = None 
    yc_label = None
    yc_prec = 3
    err = None
    datatype = None
    
    def copy(self):
        ret = MatrixWithCoords()
        ret.xc = self.xc
        ret.yc = self.yc
        ret.xc_label = self.xc_label
        ret.yc_label = self.yc_label
        ret.d = np.copy(self.d)
        ret.err = self.err
        
        return ret
    
    def expand(self, coord, axis, in_place=True):
        """ extend x or y coordinates, as specified by axis and the new coordinates, so that dissimilar
            datasets can be merged to produce a larger dataset
        """
        if axis=='x':
            shape = (len(self.yc),len(coord))
            d = np.full(shape, np.nan)
            if self.err is not None:
                err = np.full(shape, np.nan)
            else:
                err = None
            for i in range(len(self.xc)):
                i0 = np.where(coord==self.xc[i])[0][0]
                d[:,i0] = self.d[:,i]
                if err is not None:
                    err[:,i0] = self.err[:,i]
            if in_place:
                self.xc = coord
        elif axis=='y':
            shape = (len(coord),len(self.xc))
            d = np.full(shape, np.nan)
            if self.err is not None:
                err = np.full(shape, np.nan)
            else:
                err = None
            for i in range(len(self.yc)):
                i0 = np.where(coord==self.yc[i])[0][0]
                d[i0,:] = self.d[i,:]
                if err is not None:
                    err[i0,] = self.err[i,:]
            if in_place:
                self.yc = coord
        else:
            raise Exception(f"invalid axis: {axis}")
        
        if not in_place:
            return d,err

        self.d = d
        self.err = err
        
    def merge(self, ds):
        """ merge a list of MatrixWithCoord together
            all objects must have the same coordinates at least in one dimension

            since it is difficult to check whether the coordinates are the identical (prec),
            assume they are the same as long as the length are the same
        """
        common_x = True
        common_y = True
        for d in ds:
            if self.xc_label!=d.xc_label or self.yc_label!=d.yc_label:
                raise Exception("data to be merged have different x_labels.")

        ret = MatrixWithCoords()
        ret.xc = np.unique(np.hstack([m.xc for m in [self]+ds]).flatten())
        ret.yc = np.unique(np.hstack([m.yc for m in [self]+ds]).flatten())
        ret.xc_label = self.xc_label
        ret.yc_label = self.yc_label
        shape = (len(ret.yc),len(ret.xc))
                
        wt = np.zeros(shape)
        ret.d = np.zeros(shape)
        if self.err is not None:
            ret.err = np.zeros(shape)
        else:
            ret.err = None
        
        idx = None
        for d in [self]+ds:
            # expand data if necessary
            t = copy.copy(d)
            if len(ret.xc)!=len(d.xc):
                t.expand(ret.xc, "x")
            if len(ret.yc)!=len(d.yc):
                t.expand(ret.yc, "y")
            idx = ~np.isnan(t.d)
            ret.d[idx] += t.d[idx]
            if ret.err is not None:
                ret.err[idx] += t.err[idx]
            wt[idx] += 1

        idx = (wt>0)
        ret.d[idx] /= wt[idx]
        ret.d[~idx] = np.nan
        if ret.err is not None:
            ret.err[idx] /= wt[idx]
            ret.err[~idx] = np.nan

        return ret
    
    def conv(self, Nx1, Ny1, xc1, yc1, mask=None, interpolate=None, cor_factor=1, 
             inc_stat_err=False, datatype=DataType.det, err_thresh=1e-5):
        """ re-organize the 2D data based on new coordinates (xc1,yc1) for each pixel
            returns a new MatrixWithCoords, with the new coordinates specified by Nx1, Ny1
            Nx1, Ny1 can be either the number of bins or an array that specifies the bin edges
            datatype is used to describe the type of the 2D data (detector image, qr-qz map, q-phi map)
            
            err_thresh is used to weed out some outliers
        """
        ret = MatrixWithCoords()
        ret.datatype = datatype

        # correction factor is applied by dividing its value
        # should be called like this conv(...., cor_factor=exp.FSA)
        data = self.d/cor_factor
        if mask is None:
            xc1 = xc1.flatten()
            yc1 = yc1.flatten()
            data = data.flatten()
        else:
            xc1 = xc1[~(mask.map)].flatten()
            yc1 = yc1[~(mask.map)].flatten()
            data = data[~(mask.map)].flatten()            

        if hasattr(Nx1, '__iter__'): # tuple or list or np array
            Nx1 = get_bin_edges(Nx1)
            xrange = (np.min(Nx1), np.max(Nx1))
        else:
            xrange = (np.min(xc1), np.max(xc1))            
        if hasattr(Ny1, '__iter__'): # tuple or list or np array
            Ny1 = get_bin_edges(Ny1)
            yrange = (np.min(Ny1), np.max(Ny1))
        else:
            yrange = (np.min(yc1), np.max(yc1))
        xyrange = [xrange, yrange]
        
        (c_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                bins=(Nx1, Ny1), weights=np.ones(len(data)))
        (xc_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                 bins=(Nx1, Ny1), weights=xc1)
        (yc_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                 bins=(Nx1, Ny1), weights=yc1)
        cidx = (c_map>0)

        if inc_stat_err:
            dw = np.zeros_like(data)
            idx = (data!=0)    # to avoid the divergent weight for zero intensity
            dw[idx] = 1./data[idx]
            counted = np.zeros(len(data))
            counted[idx] = 1.
            (v_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                    bins=(Nx1, Ny1), weights=data*dw)
            (w_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                    bins=(Nx1, Ny1), weights=dw)
            # we are skipping the zero intensity pixels
            (c1_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                     bins=(Nx1, Ny1), weights=counted)   
            idx = (w_map>0)
            v_map[idx] /= w_map[idx]
            v_map[~idx] = np.nan
            e_map = np.zeros_like(v_map)
            e_map[idx] = 1./np.sqrt(w_map[idx])
            e_map[~idx] = np.nan
            # scale data based on how many pixels are counted vs should have been counted
            scl = np.sqrt(c_map[idx]/c1_map[idx])
            v_map[idx] /= scl
            e_map[idx] /= scl
        else:
            (v_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                    bins=(Nx1, Ny1), weights=data)
            (v2_map, x_edges, y_edges) = histogram2d(xc1, yc1, range=xyrange,
                                                     bins=(Nx1, Ny1), weights=data*data)
            v_map[cidx] /= c_map[cidx]
            v_map[~cidx] = np.nan
            v2_map[cidx] /= c_map[cidx]
            e_map = np.sqrt(v2_map-v_map*v_map)

        ret.xc = (x_edges[:-1] + x_edges[1:])/2
        ret.yc = (y_edges[:-1] + y_edges[1:])/2

        xc_map[cidx] /= c_map[cidx]
        yc_map[cidx] /= c_map[cidx]

        ret.err = e_map.T

        # re-evaluate based on the actual coordinates instead of the expected
        # unable to do it in 2D
        # default is "x" for q in q-phi conversion
        if interpolate=='x': 
            for i in range(len(ret.yc)):
                idx = np.isnan(v_map[:,i])
                if (len(v_map[:,i][~idx])>2):
                    v_map[:,i][~idx] = np.interp(ret.xc[~idx], xc_map[:,i][~idx], v_map[:,i][~idx])
        elif interpolate=='y':
            for i in range(len(ret.xc)):
                idx = np.isnan(v_map[i,:])
                if (len(v_map[i,:][~idx])>2):
                    v_map[i,:][~idx] = np.interp(ret.yc[~idx], yc_map[i,:][~idx], v_map[i,:][~idx])

        ret.d = v_map.T
        idx = (ret.d==0) | (ret.err<err_thresh)
        ret.d[idx] = np.nan
        ret.err[idx] = np.nan
        return ret

    def plot(self, ax=None, logScale=False, aspect='auto', colorbar=False, sc_factor=None, clim="auto", nolabel=False, **kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        # need to fix the direction of y-axis; in HPLC y is q and increases downward; q-phi map is the opposite 
        xx = np.tile(self.xc, [len(self.yc),1])
        if sc_factor=="x":
            sc_factor = xx
        elif sc_factor=="x2":
            sc_factor = xx*xx
        elif sc_factor=="x0.5":
            sc_factor = np.power(xx, 0.5)
        elif sc_factor=="x1.5":
            sc_factor = np.power(xx, 1.5)
        else:
            sc_factor = 1
        kwargs.pop('sc_factor', None)
            
        if clim=="auto":
            clim = auto_clim(self.d*sc_factor, logScale)
        if logScale:
            im = ax.imshow(np.log(self.d*sc_factor), aspect=aspect, clim=np.log(clim), origin="lower", **kwargs)
        else:
            im = ax.imshow(self.d*sc_factor, aspect=aspect, clim=clim, origin="lower", **kwargs)
        if not nolabel:
            ax.set_xlabel('ix')
            ax.set_ylabel('iy')
        else:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        ax.format_coord = self.format_coord
        
        if aspect=='auto':
            axx = ax.twiny()
            gpindex,gpvalues,gplabels = grid_labels(self.xc)
            axx.set_xticks(gpindex)
            if not nolabel:
                axx.set_xticklabels(gplabels)
                if self.xc_label:
                    axx.set_xlabel(self.xc_label)
            else:
                axx.xaxis.set_visible(False)

            axy = ax.twinx()
            gpindex,gpvalues,gplabels = grid_labels(self.yc)
            axy.set_yticks(gpindex)
            if not nolabel:
                axy.set_yticklabels(gplabels)
                if self.yc_label:
                    axy.set_ylabel(self.yc_label)
            else:
                axy.yaxis.set_visible(False)

            axy.format_coord = ax.format_coord #ax.format_coord #make_format(ax2, ax1)
        
        if colorbar:
            plt.colorbar(im)
        #plt.connect('button_press_event', self.mouse_press)
        #plt.connect('button_release_event', self.mouse_release)

    def format_coord(self, x, y):
        ix = x
        iy = y  
        msg = f"ix={ix:.1f}, iy={iy:.1f}; "
        
        col = int(x+0.5)
        row = int(iy+0.5)
        xc0 = self.xc[col] #np.interp(col, np.arange(len(self.xc)), self.xc)
        yc0 = self.yc[row] #np.interp(row, np.arange(len(self.yc)), self.yc)
        msg += f"{self.xc_label}={xc0:.{self.xc_prec}f}, {self.yc_label}={yc0:.{self.yc_prec}f}: "
        
        if col>=0 and col<len(self.xc) and row>=0 and row<len(self.yc):
            val = self.d[row][col]
            msg += f"{val:.2f}  "
        return msg
                
    def roi(self, x1, x2, y1, y2, mask=None):
        """ return a ROI within coordinates of x=x1~x2 and y=y1~y2 
        """
        ret = MatrixWithCoords()      
        ret.datatype = self.datatype
        
        xidx = (self.xc>=np.min([x1,x2])) & (self.xc<=np.max([x1,x2]))
        yidx = (self.yc>=np.min([y1,y2])) & (self.yc<=np.max([y1,y2]))
        t1 = np.tile(xidx, [len(yidx),1])
        t2 = np.tile(yidx, [len(xidx),1]).T

        ret.xc = self.xc[xidx]
        ret.yc = self.yc[yidx]
        ret.xc_label = self.xc_label
        ret.yc_label = self.yc_label
        ret.d = np.asarray(self.d[t1*t2].reshape((len(ret.yc),len(ret.xc))), dtype=float)
        if self.err is not None:
            ret.err = np.asarray(self.err[t1*t2].reshape((len(ret.yc),len(ret.xc))), dtype=float)
        
        if mask is not None:
            idx = mask.map[t1*t2].reshape((len(ret.yc),len(ret.xc)))
            ret.d[idx] = np.nan
        
        return ret

    def val(self, x0, y0, dx, dy, mask=None):
        """ return the averaged value of the data at coordinates (x0,y0), within a box of (dx, dy) 
        """
        roi = self.roi(x0-0.5*dx, x0+0.5*dx, y0-0.5*dy, y0+0.5*dy, mask=mask)
        d = roi.d[~np.isnan(roi.d)]
        return np.sum(d)/len(d)
    
    def line_profile(self, direction, xrange=None, yrange=None, plot_data=False, ax=None, **kwargs):
        """ return the line profile along the specified direction ("x" or "y"), 
            within the range of coordinate given by crange=[min, max] in the other direction
        """
        if xrange is None:
            xrange = [self.xc[0], self.xc[-1]]
        if yrange is None:
            yrange = [self.yc[0], self.yc[-1]]
        if direction=="x":
            cc = self.xc[(self.xc>=xrange[0]) & (self.xc<=xrange[1])]
        elif direction=="y":
            cc = self.yc[(self.yc>=yrange[0]) & (self.yc<=yrange[1])]
        else:
            raise exception(f"invalid direction: {direction}")
            
        roi = self.roi(xrange[0], xrange[1], yrange[0], yrange[1])
        dd,ee = roi.flatten(axis=direction)
        
        if plot_data:
            if ax is None:
                fig,ax = plt.figure()
            ax.plot(cc,dd, **kwargs)

        return cc,dd,ee
    
    def flatten(self, axis='x', method='simple'):
        """ collapse the matrix onto the specified axis and turn it into an array 
        """
        if not axis in ['x', 'y']:
            raise Exception(f"unkown axis for flattening data: {axis}")
        if not method in ['simple', 'err_weighted']:
            raise Exception(f"unknow method for averaging: {method}")
        if method=="err_weighted" and self.err is None:
            raise Exception(f"err_weighted averaging requires error bar data.")
    
        dat = []
        err = []
        if axis=='x':
            for i in range(len(self.yc)):
                dat.append(self.d[i,:])
                if self.err is not None:
                    err.append(self.err[i,:])
            dd,ee = calc_avg(dat, err, method=method)
        if axis=='y': 
            for i in range(len(self.xc)):
                dat.append(self.d[:,i])
                if self.err is not None:
                    err.append(self.err[:,i])
            dd,ee = calc_avg(dat, err, method=method)
        
        return dd,ee

    def average(self, datalist:list, weighted=False):
        """ average with the list of data given
            all data must have the same coordinates and datatype
        """
        ret = copy.deepcopy(self)
        for d in datalist:
            if not np.array_equal(ret.xc, d.xc) or not np.array_equal(ret.xc, d.xc) or ret.datatype!=d.datatype:
                raise Exception("attempted average between incompatiple data.")
            if weighted and d.err is None:
                raise Exception("weighted average requires error bars for each dataset.")

        dat = [d.d for d in [self]+datalist]
        if weighted:
            err = [d.err for d in [self]+datalist]
            calc_avg(dat, err, "")
        else: 
            calc_avg(dat, [])
            
        return ret
    
    def bkg_cor(self, dbkg, scale_factor=1.0, in_place=True):
        if not np.array_equal(self.xc, dbkg.xc) or not np.array_equal(self.xc, dbkg.xc) or self.datatype!=dbkg.datatype:
            raise Exception("attempted background subtraction using imcompatiple data.")
        if in_place:
            self.d -= np.asarray(dbkg.d*scale_factor, dtype=self.d.dtype)
            if self.err and dbkg.err:
                self.err += dbkg.err*scale_factor
        else:
            ret = copy.deepcopy(self)
            ret.d = self.d - dbkg.d*scale_factor
            if self.err and dbkg.err:
                ret.err = self.err + dbkg.err*scale_factor
        
    def apply_symmetry(self):
        """ this only applies if the y coordinate is angle and covers 360deg
        """
        t = self.copy()
        Np = int(len(self.yc)/2)
        t.d = np.vstack([self.d[Np:,:], self.d[:Np,:]])
        return self.merge([t])

    def fill_gap(self, method="spline", param=0.05):
        """ 
            for now this is interepolating in x coordinate only 
            
            interpolate within each row 
            methods should be "linear" or "spline"

            a better version of this should use 2d interpolation
            but only fill in the space that is narrow enough in one direction (e.g. <5 missing data points)
        """
        h,w = self.d.shape
        ret = self.copy()

        xx1 = np.arange(w)
        for k in range(h):
            yy1 = self.d[k,:]    
            idx = ~np.isnan(yy1)
            if len(idx)<=10:  # too few valid data points
                continue
            idx1 = np.where(idx)[0]
            # only need to refill the values that are currently nan
            if len(idx1)>1:
                idx2 = np.copy(idx)
                idx2[:idx1[0]] = True
                idx2[idx1[-1]:] = True
                if method=="linear":
                    ret.d[k,~idx2] = np.interp(xx1[~idx2], xx1[idx], yy1[idx])
                elif method=="spline":
                    fs = uspline(xx1[idx], yy1[idx])
                    fs.set_smoothing_factor(param)
                    ret.d[k,~idx2] = fs(xx1[~idx2])
                else:
                    raise Exception(f"unknown method for intepolation: {method}")

        return ret
                
                
                
def flip_array(d, flip):
    if flip == 0:
        return d
    df = copy.copy(d)
    if flip<0:
        df = np.fliplr(df)
        flip = -flip
    df = np.rot90(df, flip)
        
    return df

def unflip_array(d, flip):
    if flip==0:
        return d
    df = copy.copy(d)
    df = np.rot90(df, 4-abs(flip))  # rotate -90
    if flip<0:
        df = np.fliplr(df)
        
    return df


class Data2d:
    """ 2D scattering data class
        stores the scattering pattern itself, 
    """

    def __init__(self, img, timestamp=None, uid='', exp=None, ignore_flip=False, label='', dtype=None, flat=None):
        """ read 2D scattering pattern
            img can be either a filename (rely on Fabio to deal with the file format) or a numpy array 
        """
        self.exp = None
        self.timestamp = None
        self.uid = None
        self.data = MatrixWithCoords()
        self.qrqz_data = MatrixWithCoords()
        self.qphi_data = MatrixWithCoords()
        self.label = label
        self.md = {}
        
        if isinstance(img, str):
            f = fabio.open(img)
            self.im = np.asarray(f.data)
            self.label = os.path.basename(img)
            # get other useful information from the header
            # cbf header 
            if '_array_data.header_contents' in f.header.keys():
                if f.header['_array_data.header_convention']=='PILATUS_1.2':
                    ts = f.header['_array_data.header_contents'].split("# ")[2]
                    self.timestamp = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f\r")
                    try:
                        self.uid = f.header['_array_data.header_contents'].split("# ")[16].split("uid=")[1].rstrip("\r\n")
                    except:
                        self.uid = ''
            #f.close()
        elif isinstance(img, np.ndarray): 
            self.im = img
            self.timestamp = timestamp
            self.uid = uid
        else:
            raise Exception('Not sure how to create Data2d from img ...')

        if dtype is not None:
            self.im = np.asarray(self.im, dtype=dtype)
            
        # self.im always stores the original image
        # self.data store the array data after the flip operation
        if exp is not None:
            self.set_exp_para(exp, ignore_flip)
        else:
            # temporarily set data to img, without a defined exp_para
            self.flip(0)
            self.height, self.width = self.data.d.shape

    def save_as_8bit_tif(self, filename, mask=None, max_intensity=2000):
        """ the purpose of this function is to create a image from which a new mask can be created
            8bit depth is sufficient for this purpose and log scale is used 
            best data for this purpose is water scattering or empty cell scattering
        """
        img_data = np.log(self.data.d+1)+1
        idx = np.isinf(img_data)|np.isnan(img_data)
        img_data *= 240/np.log(max_intensity)
        img_data[idx] = 255
        if mask is not None:
            img_data[mask.map] = 255
        im = Image.fromarray(np.uint8(img_data))
        im.save(filename)    
            
    def set_timestamp(self, ts):
        # ts must be a valid datetime structure
        self.timestamp = ts

    def flip(self, flip):
        """ this is a little complicated
            if flip<0, do a mirror operation first
            the absolute value of flip is the number of 90-deg rotations
        """  
        self.data.d = flip_array(self.im, flip)

    def set_exp_para(self, exp, ignore_flip=False):
        """ ignore_flip is useful if the flipped image is saved
        """
        if ignore_flip:
            self.flip(0)
        else:
            self.flip(exp.flip)
        (self.height, self.width) = np.shape(self.data.d)
        self.exp = exp
        if exp.ImageHeight!=self.height or exp.ImageWidth!=self.width:
            raise Exception('mismatched shape between the data (%d,%d) and ExpPara (%d,%d).' % 
                            (self.width, self.height, exp.ImageWidth, exp.ImageHeight)) 
        self.data.xc = np.arange(self.width)
        self.data.yc = np.flipud(np.arange(self.height)) 
        self.data.datatype = DataType.det

    def conv_Iqrqz(self, Nqr, Nqz, mask=None, cor_factor=1):
        self.qrqz_data = self.data.conv(Nqr, Nqz, self.exp.Qr, self.exp.Qn, 
                                        mask=mask, cor_factor=cor_factor, datatype=DataType.qrqz)
        
    def conv_Iqphi(self, Nq, Nphi, mask=None, cor_factor=1, regularize=True):
        """ regularize = True:
            for convinience of looking up the intensity at (q, phi+180) (for the purpose of filling 
            up missing intensity based on centrosymmetry), the grid for phi should be specified, such
            that phi+180 (fold into [-180, 180] if necessary) is also in the array
        """
        if regularize:
            phi_max = self.exp.Phi.max()
            phi_min = self.exp.Phi.min()
            Nphi1 = int(360./(phi_max-phi_min)*Nphi/2+0.5)
            phi = np.linspace(-180., 180, Nphi1*2+1)         # makes sure that 0 is at the center of a bin
            Nphi = phi[(phi>=phi_min) & (phi<phi_max)]     # replace the number of bins with edges
        self.qphi_data = self.data.conv(Nq, Nphi, self.exp.Q, self.exp.Phi, 
                                        mask=mask, cor_factor=cor_factor, datatype=DataType.qphi)
        
    def conv_Iq(self, qgrid, mask=None, cor_factor=1, 
                adjust_edges=True, interpolate=True, min_norm_scale=0.002):
        
        dd = self.data.d/cor_factor
        
        # Pilatus might use negative values to mark dead pixels
        idx = (self.data.d>=0)
        if mask is not None:
            idx &= ~(mask.map)
            
        qd = self.exp.Q[idx].flatten()
        dd = np.asarray(dd[idx].flatten(), dtype=float)  # other wise dd*dd might become negative
        
        if adjust_edges:
            # willing to throw out some data, but the edges strictly correspond to qgrid 
            dq  = qgrid[1:]-qgrid[:-1]
            dq1 = np.hstack(([dq[0]], dq))

            bins = [qgrid[0]-dq1[0]/2]
            bidx = []
            binw = []

            for i in range(len(qgrid)):
                el = qgrid[i] - dq1[i]/2
                eu = qgrid[i] + dq1[i]/2
                if i==0 or np.fabs(el-bins[-1])<dq1[i]/100:
                    bins += [eu]
                    bidx += [True]
                    binw += [dq1[i]]
                else:
                    bins += [el, eu]
                    bidx += [False, True]
                    binw += [dq1[i-1], dq1[i]]
        else:
            # keep all the data, but the histogrammed data will be less accurate 
            bins = np.append([2*qgrid[0]-qgrid[1]], qgrid) 
            bins += np.append(qgrid , [2*qgrid[-1]-qgrid[-2]])
            bins *= 0.5
            bidx = np.ones(len(qgrid), dtype=bool)

        norm = np.histogram(qd, bins=bins, weights=np.ones(len(qd)))[0][bidx] 
        qq = np.histogram(qd, bins=bins, weights=qd)[0][bidx]
        Iq = np.histogram(qd, bins=bins, weights=dd)[0][bidx]
        Iq2 = np.histogram(qd, bins=bins, weights=dd*dd)[0][bidx]

        idx1 = (norm>min_norm_scale*np.arange(len(norm))**2)
        qq[idx1] /= norm[idx1]
        Iq[idx1] /= norm[idx1]
        Iq2[idx1] /= norm[idx1]
        dI = Iq2-Iq*Iq
        dI[dI<0] = 0  # sometimes get runtime warnings
        dI = np.sqrt(dI)
        dI[idx1] /= np.sqrt(norm[idx1])
        qq[~idx1] = np.nan
        Iq[~idx1] = np.nan
        dI[~idx1] = np.nan
        
        if interpolate:
            Iq = np.interp(qgrid, qq, Iq)
        
        return Iq,dI

    
