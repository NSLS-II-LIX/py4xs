import numpy as np
import fabio
import datetime,os,copy
from py4xs.mask import Mask
from py4xs.local import ExpPara
from py4xs.utils import calc_avg
import pylab as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from enum import Enum 
from PIL import Image

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
        gpindex = np.linspace(1, len(grid), N, dtype=np.int)-1
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
    
    return gpindex,gpvalues,gplabels

class MatrixWithCoords:
    # 2D data with coordinates
    d = None
    xc = None
    xc_label = None
    yc = None 
    yc_label = None
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
        
        #print(common_x,common_y,len(ret.xc),len(ret.yc),shape)
        
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
    
    def conv0(self, Nx1, Ny1, xc1, yc1, mask=None, cor_factor=1, datatype=DataType.det):
        """ re-organize the 2D data based on new coordinates (xc1,yc1) for each pixel
            returns a new MatrixWithCoords, with the new coordinates specified by Nx1, Ny1
            Nx1, Ny1 can be either the number of bins or an array that specifies the bin edges
            datatype is used to describe the type of the 2D data (detector image, qr-qz map, q-phi map)
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
        if hasattr(Ny1, '__iter__'): # tuple or list or np array
            Ny1 = get_bin_edges(Ny1)

        (v_map, x_edges, y_edges) = np.histogram2d(xc1, yc1,
                                                bins=(Nx1, Ny1), weights=data)
        (c_map, x_edges, y_edges) = np.histogram2d(xc1, yc1,
                                                bins=(Nx1, Ny1), weights=np.ones(len(data)))

        idx = (c_map<=0) # no data
        c_map[idx] = 1.
        v_map[idx] = np.nan
        ret.d = np.fliplr(v_map/c_map).T
        ret.xc = (x_edges[:-1] + x_edges[1:])/2
        ret.yc = (y_edges[:-1] + y_edges[1:])/2
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
        if hasattr(Ny1, '__iter__'): # tuple or list or np array
            Ny1 = get_bin_edges(Ny1)

        (c_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=np.ones(len(data)))
        (xc_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=xc1)
        (yc_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=yc1)
        cidx = (c_map>0)

        if inc_stat_err:
            dw = np.zeros_like(data)
            idx = (data!=0)    # to avoid the divergent weight for zero intensity
            dw[idx] = 1./data[idx]
            counted = np.zeros(len(data))
            counted[idx] = 1.
            (v_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=data*dw)
            (w_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=dw)
            # we are skipping the zero intensity pixels
            (c1_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=counted)   
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
            (v_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=data)
            (v2_map, x_edges, y_edges) = np.histogram2d(xc1, yc1, bins=(Nx1, Ny1), weights=data*data)
            v_map[cidx] /= c_map[cidx]
            v_map[~cidx] = np.nan
            v2_map[cidx] /= c_map[cidx]
            e_map = np.sqrt(v2_map-v_map*v_map)

        ret.xc = (x_edges[:-1] + x_edges[1:])/2
        ret.yc = (y_edges[:-1] + y_edges[1:])/2

        xc_map[cidx] /= c_map[cidx]
        yc_map[cidx] /= c_map[cidx]

        ret.err = np.fliplr(e_map).T

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

        ret.d = np.fliplr(v_map).T
        idx = (ret.d==0) | (ret.err<err_thresh)
        ret.d[idx] = np.nan
        ret.err[idx] = np.nan
        return ret

    def plot(self, ax=None, logScale=False, **kwargs):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        if logScale:
            ax.imshow(np.log(self.d), **kwargs)
        else:
            ax.imshow(self.d, **kwargs)
        ax.set_xlabel('ix')
        ax.set_ylabel('iy')

        axx = ax.twiny()
        gpindex,gpvalues,gplabels = grid_labels(self.xc)
        axx.set_xticks(gpindex)
        axx.set_xticklabels(gplabels)
        if self.xc_label:
            axx.set_xlabel(self.xc_label)

        axy = ax.twinx()
        gpindex,gpvalues,gplabels = grid_labels(self.yc)
        axy.set_yticks(gpindex)
        axy.set_yticklabels(gplabels)
        if self.yc_label:
            axy.set_ylabel(self.yc_label)
            
        axy.format_coord = ax.format_coord #make_format(ax2, ax1)

    def roi(self, x1, x2, y1, y2, mask=None):
        """ return a ROI within coordinates of x=x1~x2 and y=y1~y2 
        """
        ret = MatrixWithCoords()      
        ret.datatype = self.datatype
        
        xidx = (self.xc>=np.min([x1,x2])) & (self.xc<=np.max([x1,x2]))
        yidx = (self.yc>=np.min([y1,y2])) & (self.yc<=np.max([y1,y2]))
        t1 = np.tile(xidx, [len(yidx),1])
        t2 = np.tile(np.flipud(yidx), [len(xidx),1]).T

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
        d = roi[~np.isnan(roi)]
        return np.sum(d)/len(d)
    
    def line_cut(self, x0, y0, ang, lT, lN, nT, nN, mask=None):
        """ take a line cut
                (x0, y0): center
                ang: orientation (in degrees) 
                lT, lN: half length (tangential) and half width (normal) of the cut in pixels 
                lN, nN: number of returned data points logn the legnth and width of the cut
        """
        ret = MatrixWithCoords()
        ret.datatype = self.datatype
        ang = np.radians(ang)
        
        if self.yc[0]>self.yc[-1]:  # raw data, y coordinate lower at the top of the image
            ang = -ang
        
        # xc/yc need to have the same dimension as d
        (h,w) = self.d.shape
        xc = np.repeat(self.xc, h).reshape((w, h)).T
        yc = np.repeat(np.flipud(self.yc), w).reshape((h, w))
        
        distT =  (xc.flatten()-x0)*np.cos(ang) + (yc.flatten()-y0)*np.sin(ang) 
        distN = -(xc.flatten()-x0)*np.sin(ang) + (yc.flatten()-y0)*np.cos(ang) 

        bin_t = np.linspace(-lT, lT, nT)
        bin_n = np.linspace(-lN, lN, nN)

        # exclude masked pixels or pixels that contain non-numerical values
        if mask is None:
            idx = np.isnan(self.d)
        else:
            idx = mask.map | np.isnan(self.d)
        idx = ~idx.flatten()
        
        (v_map, t_edges, n_edges) = np.histogram2d(distT[idx], distN[idx],
                                                bins=(bin_t, bin_n), weights=self.d.flatten()[idx])
        (c_map, t_edges, n_edges) = np.histogram2d(distT[idx], distN[idx],
                                                bins=(bin_t, bin_n), weights=np.ones(len(distT[idx])))
                                                 
        ret.d = np.fliplr(v_map/c_map).T    
        ret.xc = (t_edges[:-1] + t_edges[1:])/2
        ret.yc = (n_edges[:-1] + n_edges[1:])/2
        
        if self.yc.flatten()[0]>self.yc.flatten()[-1]:  # raw data, y coordinate lower at the top of the image
            ret.d = np.flipud(ret.d)
        
        return ret
    
    def flatten(self, axis='x', method='err_weighted'):
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
        if axis=='y':
            for i in range(len(self.xc)):
                dat.append(self.d[:,i])
                if self.err is not None:
                    err.append(self.err[:,i])
        if self.err is None:
            err = None
        dd,ee = calc_avg(dat, err, method=method)
        
        return dd,ee

    def average(self, datalist:list, weighted=False):
        """ average with the list of data given
            all data must have the same coordinates and datatype
        """
        for d in datalist:
            if not np.array_equal(ret.xc, d.xc) or not np.array_equal(ret.xc, d.xc) or ret.datatype!=d.datatype:
                raise Exception("attempted average between incompatiple data.")
            if weighted and d.err is None:
                raise Exception("weighted average requires error bars for each dataset.")

        ret = copy.deepcopy(self)
        dat = [d.d for d in [self]+datalist]
        if weighted:
            err = [d.err for d in [self]+datalist]
            calc_avg(dat, err, "")
        else: 
            calc_avg(dat)
            
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


class Data2d:
    """ 2D scattering data class
        stores the scattering pattern itself, 
    """

    def __init__(self, img, timestamp=None, uid='', exp=None, label='', dtype=None, flat=None):
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
            self.set_exp_para(exp)
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
        self.data.d = self.im.copy()
        if flip == 0:
            return
        if flip<0:
            self.data.d = np.fliplr(self.data.d)
            flip = -flip
        for _ in range(flip):
            self.data.d = np.rot90(self.data.d)

    def set_exp_para(self, exp):
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
        dd = np.asarray(dd[idx].flatten(), dtype=np.float)  # other wise dd*dd might become negative
        
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

    
class Axes2dPlot:
    """
        display MatrixWithCoordinates, image origin at the upper left corner
        option to provide coordinates translation
    """    
    def __init__(self, ax, data, exp=None, datatype=DataType.det):
        """
        """
        self.ax = ax
        self.cmap = plt.get_cmap()
        self.scale = 'linear'
        self.d2 = data
        self.ptns = []
        self.xlim = None
        self.ylim = None
        self.img = None
        self.coordinate_translation = None
        self.exp = exp
        self.n_step = 121
        self.datatype = datatype
        self.capture_mouse()

    def capture_mouse(self):
        self.ax.figure.canvas.mpl_connect('button_press_event', self.clicked)
        # self.ax.figure.canvas.mpl_connect('motion_notify_event', self.move_event)

    def right_click(self, event):
        """ display menu to change image color scale etc.
        """
        pass

    def clicked(self, event):
        if event.inaxes != self.ax:
            return True
        x = event.xdatta 
        y = event.ydata 
        if self.coordinate_translation=="xy2qrqz":
            (q, phi, qr, qz) = self.exp.calc_from_XY(np.asarray([x]),np.asarray([y]))
            msg = "qr=%.4f , qz=%.4f" % (qr[0], qz[0])
        elif self.coordinate_translation=="xy2qphi":
            (q, phi, qr, qz) = self.exp.calc_from_XY(np.asarray([x]),np.asarray([y]))
            msg = "q=%.4f, phi=%.1f" % (q[0],phi[0])
        else:
            msg = f"({x:.1f}, {y:.1f})"
        self.ax.set_title(msg, fontsize="small")
        print(msg)
        return True

    # best modify d2.xc/d2.yc instead of using xscale/yscale
    def plot(self, showMask=None, logScale=False, mask_alpha=0.1,
             aspect=1., xscale=1., yscale=1.):

        dd = np.asarray(self.d2.d, dtype=np.float)

        immax = np.average(dd) + 5 * np.std(dd)
        immin = np.average(dd) - 5 * np.std(dd)
        if immin < 0:
            immin = 0

        if showMask and self.exp:
            self.ax.imshow(self.exp.mask.map, cmap="gray")
        if logScale:
            self.img = self.ax.imshow(dd, aspect=aspect, alpha=1-mask_alpha,
                                      cmap=self.cmap, interpolation='nearest', norm=LogNorm(),
                                      extent=[self.d2.xc[0]*xscale, self.d2.xc[-1]*xscale, 
                                              self.d2.yc[0]*yscale, self.d2.yc[-1]*yscale])
        else:
            self.img = self.ax.imshow(dd, vmax=immax, vmin=immin, 
                                      aspect=aspect, alpha=1-mask_alpha,
                                      cmap=self.cmap, interpolation='nearest',
                                      extent=[self.d2.xc[0]*xscale, self.d2.xc[-1]*xscale, 
                                              self.d2.yc[0]*yscale, self.d2.yc[-1]*yscale]) 
        self.xlim = [self.d2.xc[0]*xscale, self.d2.xc[-1]*xscale]
        self.ylim = [self.d2.yc[0]*yscale, self.d2.yc[-1]*yscale]

    def set_color_scale(self, cmap, gamma=1):
        """ linear, log/gamma
        """
        if not gamma == 1:
            cmap = cmap_map(lambda x: np.exp(gamma * np.log(x)), cmap)
        self.cmap = cmap
        if self.img is not None:
            self.img.set_cmap(cmap)

    # xvalues and yvalues should be arrays of the same length, positons to be markes
    # used to mark a pixel position, in all types of 2D maps, or a (qr,qz) or (p,phi) pair on the detector image
    def mark_points(self, xvalues, yvalues, fmt="r+", datatype=None):
        if self.d2.datatype==DataType.det and self.exp!=None:
            (Q, Phi, Qr, Qn) = self.exp.calc_from_XY(np.asarray(xvalues), np.asarray(yvalues))
            if datatype==DataType.qrqz:
                (xvalues,yvalues) = (Qr,Qn)
            elif datatype==DataType.qphi:
                (xvalues,yvalues) = (Q,Phi)
        elif datatype==DataType.det and self.exp!=None:
            if self.d2.datatype==DataType.qrqz:
                (xvalues,yvalues) = self.exp.calc_from_QrQn(np.asarray(xvalues), np.asarray(yvalues))
            elif self.d2.datatype==DataType.qphi: # need to convert angular unit
                (xvalues,yvalues) = self.exp.calc_from_QPhi(np.asarray(xvalues), np.asarray(yvalues))
        elif self.d2.datatype!=datatype and datatype!=None:
            raise Exception("imcompatible data types:", datatype, self.d2.datatype)
            
        ptn = [[xvalues, yvalues], fmt]    
        self.ptns.append(ptn)
        self.draw_dec()
    
    # xvalues and yvalues can have different lengths 
    def mark_coords(self, xvalues, yvalues=None, fmt="r-", datatype=None):
        if datatype==None or datatype==self.d2.datatype: # straight lines between max/min
            for xv in xvalues:
                self.ptns.append([[[xv, xv], self.ylim], fmt])
            for yv in yvalues:
                self.ptns.append([[self.xlim, [yv, yv]], fmt])
        elif self.exp==None:
            raise Exception("undefined ExpPara for the plot.")
        elif datatype==DataType.q:
            phi = np.linspace(0, 360., self.n_step)
            if self.d2.datatype==DataType.det:
                for q in xvalues:
                    (xv, yv) = self.exp.calc_from_QPhi(q*np.ones(self.n_step), phi)
                    self.ptns.append([[xv, yv], fmt])
            elif self.d2.datatype==DataType.qrqz:
                for q in xvalues:
                    self.ptns.append([[q*np.cos(np.radians(phi)), q*np.sin(np.radians(phi))], fmt])
            elif self.d2.datatype==DataType.qphi:
                for q in xvalues:
                    self.ptns.append([[[q, q], self.ylim], fmt])
        elif self.d2.datatype==DataType.det and datatype==DataType.qrqz:
            qr_grid = np.linspace(self.exp.Qr.min(), self.exp.Qr.max(), self.n_step)
            qz_grid = np.linspace(self.exp.Qn.min(), self.exp.Qn.max(), self.n_step)
            for qr in xvalues:
                (xv,yv) = self.exp.calc_from_QrQn(qr*np.ones(self.n_step), qz_grid)
                self.ptns.append([[xv, yv], fmt])
            for qz in yvalues:
                (xv,yv) = self.exp.calc_from_QrQn(qr_grid, qz*np.ones(self.n_step))
                self.ptns.append([[xv, yv], fmt])
        elif self.d2.datatype==DataType.det and datatype==DataType.qphi:
            q_grid = np.linspace(self.exp.Q.min(), self.exp.Q.max(), self.n_step)
            phi_grid = np.linspace(self.exp.Phi.min(), self.exp.Phi.max(), self.n_step)
            for q in xvalues:
                (xv,yv) = self.exp.calc_from_QPhi(q*np.ones(self.n_step), phi_grid)
                self.ptns.append([[xv, yv], fmt])            
            for phi in yvalues:
                (xv,yv) = self.exp.calc_from_QPhi(q_grid, phi*np.ones(self.n_step))
                self.ptns.append([[xv, yv], fmt])
        else:
            raise Exception("imcompatible data types:", datatype, self.d2.datatype)
            
        self.draw_dec()            
    
    def mark_standard(self, std, sym="k:"):
        """ std should be one of these: 
            AgBH: mutiples of 0.1076, then 1.37
            sucrose: 0.5933, 0.8289, 0.9054, 0.9336, 1.1000
            CeO2: 2.0116, 1.8985, 3.2838, 3.8504
            LaB6: 1.5115, 2.1376, 2.6180, 3.0230, 3.3799, 3.7025
        """
        if std=="AgBH":
            q_std = np.hstack((0.1076*np.arange(1,10), [1.37]))
        elif std=="sucrose":
            q_std = [0.5933, 0.8289, 0.9054, 0.9336, 1.1000]
        elif std=="CeO2": # http://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nbsmonograph25-20.pdf
            q_std = [ 2.0116, 2.3219, 3.2838, 3.8504 ]
        elif std=="LaB6": # NIST SRM 660c
            q_std = [ 1.5115, 2.1376, 2.6180, 3.0230, 3.3799, 3.7025 ]
        elif isinstance(std, list):
            q_std = np.asarray(std, dtype=np.float)
        else: # unknown standard
            return 
        self.mark_coords(q_std, [], sym, DataType.q)
        if self.exp and self.datatype==DataType.det:
            self.mark_points([self.exp.bm_ctr_x], [self.exp.bm_ctr_y], datatype=DataType.det)
    
    # simply plot the x,y coordinates
    def mark_line(self, xv, yv, fmt="r-"):
        self.ptns.append([[xv, yv], fmt])
        self.draw_dec()
    
    # ptn = [data, fmt]
    # 
    def draw_dec(self):
        for ptn in self.ptns:
            ([px, py], fmt) = ptn
            self.ax.plot(px, py, fmt)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        

