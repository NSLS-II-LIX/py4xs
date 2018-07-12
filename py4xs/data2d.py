import numpy as np
import fabio
import datetime
from py4xs.mask import Mask
import pylab as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from enum import Enum 
import copy


class DataType(Enum):
    det = 1
    qrqz = 2
    qphi = 3
    q = 4

    
def RotationMatrix(axis, angle):
    if axis=='x' or axis=='X':
        rot = np.asarray(
            [[1., 0., 0.],
             [0., np.cos(angle), -np.sin(angle)],
             [0., np.sin(angle),  np.cos(angle)]])
    elif axis=='y' or axis=='Y':
        rot = np.asarray(
            [[ np.cos(angle), 0., np.sin(angle)],
             [0., 1., 0.],
             [-np.sin(angle), 0., np.cos(angle)]])
    elif axis=='z' or axis=='Z':
        rot = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.],
             [np.sin(angle),  np.cos(angle), 0.],
             [0., 0., 1.]])
    else:
        raise ValueError('unknown axis %s' % axis)
    
    return rot 


class ExpPara:
    """
    The geomatric claculations used here are described in Yang, J Synch Rad (2013) 20, 211–218
    
    Initilized with image size:
    __init__(self, width, height)
    
    Calculate all coordinates and correction factors for each pixel:
    init_coordinates(self)
    
    Functions that can be called for converting coordinates (inputs are arrays)
    calc_from_XY(self, X, Y, calc_cor_factors=False)
    calc_from_QrQn(self, Qr, Qn, flag=False)
    calc_from_QPhi(self, Q, Phi)
    
    unit for angles is degrees
    
    """
    wavelength = None
    bm_ctr_x = None
    bm_ctr_y = None
    ratioDw = None
    grazing_incident = False
    flip = 0
    incident_angle = 0.2
    sample_normal = 0.
    rot_matrix = None
    
    def __init__(self, width, height):
        """
        define image size here, since the coordinates need to be defined later 
        might as well initilize the mask here since it has to be the same size
        """
        self.ImageWidth = width
        self.ImageHeight = height
        self.mask = Mask(width, height)
        
    def init_coordinates(self):
        """
        calculate all coordinates (pixel position as well as various derived values)
        all coordinates are stored in 2D arrays, as is the data itself in Data2D
        """
        (w,h) = (self.ImageWidth, self.ImageHeight)
        self.X = np.repeat(np.arange(w), h).reshape((w, h)).T
        X = self.X.flatten()
        Y = np.repeat(np.arange(h), w)
        self.Y = Y.reshape((h, w))
        
        (Q, Phi, Qr, Qn, FPol, FSA) = self.calc_from_XY(X, Y, calc_cor_factors=True)

        # this is the direct/shortest distance between the origin/sample and the detector, in pixels        
        self.Dd = self.ratioDw*self.ImageWidth*np.dot(np.dot(self.rot_matrix, np.asarray([0, 0, 1.])), 
                                                      np.asarray([0, 0, 1.]))        

        self.Q = Q.reshape((h, w))
        self.Qr = Qr.reshape((h, w))
        self.Qn = Qn.reshape((h, w))
        self.Phi = Phi.reshape((h, w))
        self.FPol = FPol.reshape((h, w))
        self.FSA = FSA.reshape((h, w))
        
    def calc_from_XY(self, X, Y, calc_cor_factors=False):
        """
        calculate Q values from pixel positions X and Y
        X and Y are 1D arrays
        returns reciprocal/angular coordinates, optionally returns 
        always calculates Qr and Qn, therefore incident_angle needs to be set 
        Note that angular values are in radians internally, in degrees externally 
        """
        if self.rot_matrix is None:
            raise ValueError('the rotation matrix is not yet set.')

        # the position vectors for each pixel, origin at the postion of beam impact
        # R.shape should be (3, w*h), but R.T is more convinient for matrix calculation
        # RT.T[i] is a vector
        RT = np.vstack((X - self.bm_ctr_x, -(Y - self.bm_ctr_y), 0.*X))
        
        dr = self.ratioDw*self.ImageWidth
        # position vectors in lab coordinates, sample at the origin
        [X1, Y1, Z1] = np.dot(self.rot_matrix, RT)
        Z1 -= dr

        # angles
        r3sq = X1*X1+Y1*Y1+Z1*Z1
        r3 = np.sqrt(r3sq)
        r2 = np.sqrt(X1*X1+Y1*Y1)
        Theta = 0.5*np.arcsin(r2/r3)
        Phi = np.arctan2(Y1, X1) + np.radians(self.sample_normal)

        # this is for calculating FSA
        cosTH = np.dot( np.dot(self.rot_matrix, np.asarray([0, 0, 1.])),  
                        np.vstack((X1, Y1, Z1)) ) 
        cosTH = np.fabs(cosTH) / r3

        Q = 4.0*np.pi/self.wavelength*np.sin(Theta)

        # lab coordinates
        Qz = Q*np.sin(Theta)
        Qy = Q*np.cos(Theta)*np.sin(Phi)
        Qx = Q*np.cos(Theta)*np.cos(Phi)

        # convert to sample coordinates
        alpha = np.radians(self.incident_angle)
        Qn = Qy*np.cos(alpha) + Qz*np.sin(alpha)
        Qr = np.sqrt(Q*Q-Qn*Qn)*np.sign(Qx)
        
        if calc_cor_factors==True:
            FPol = (Y1*Y1+Z1*Z1)/r3sq
            FSA = np.power(cosTH, 3)
            return (Q, np.degrees(Phi), Qr, Qn, FPol, FSA)
        else:
            return (Q, np.degrees(Phi), Qr, Qn)
    
    def calc_from_QrQn(self, Qr, Qn, flag=False):
        """
        Qr and Qn are 1D arrays 
        when flag is True, substitue Qr with the minimal Qr value at the given Qz allowed by scattering geometry
        returns the pixel positions corresponding to (Qr, Qn)
        note that the return arrays may contain non-numerical values
        """
        if self.rot_matrix is None:
            raise ValueError('the rotation matrix is not yet set.')
                    
        alpha = np.radians(self.incident_angle)
        
        if flag is True:
            k = 2.0*np.pi/self.wavelength
            tt = Qn/k -np.sin(alpha)
            Qr0 = np.empty(len(Qr))
            Qr0[tt<=1.] = np.fabs(np.sqrt(1.-(tt*tt)[tt<=1.]) - np.cos(alpha))*k
            idx1 = (Qr<Qr0) & (tt<=1.)
            Qr[idx1] = Qr0[idx1]*np.sign(Qr[idx1])

        Q = np.sqrt(Qr*Qr+Qn*Qn)
        Phi = np.empty(len(Q))

        Theta = self.wavelength*Q/(4.0*np.pi)
        idx = (Theta<=1.0)
        Theta = np.arcsin(Theta[idx])
        Phi[~idx] = np.nan
        
        Qz = Q[idx]*np.sin(Theta)
        Qy = (Qn[idx] - Qz*np.sin(alpha)) / np.cos(alpha)
        tt = Q[idx]*Q[idx] - Qz*Qz - Qy*Qy
        idx2 = (tt>=0)
        Qx = np.empty(len(Q[idx]))
        Qx[idx2] = np.sqrt(tt[idx2])*np.sign(Qr[idx][idx2])
        
        Phi[idx & idx2] = np.arctan2(Qy[idx2], Qx[idx2]) 
        Phi[idx & ~idx2] = np.nan

        return self.calc_from_QPhi(Q, np.degrees(Phi))    

    def calc_from_QPhi(self, Q, Phi):
        """
        Q and Phi are 1D arrays 
        Phi=0 is the y-axis (pointing up), unless sample_normal is non-zero
        returns the pixel positions corresponding to (Q, Phi)
        note that the return arrays may contain non-numerical values
        """
        if self.rot_matrix is None:
            raise ValueError('the rotation matrix is not yet set.')
                
        Theta = self.wavelength*Q/(4.0*np.pi)
        X0 = np.empty(len(Theta))
        Y0 = np.empty(len(Theta))
        
        idx = (Theta<=1.0) & (~np.isnan(Phi))  # Phi might contain nan from calc_from_QrQn()
        Theta = np.arcsin(Theta[idx]);
        
        Phi = np.radians(Phi[idx]) - np.radians(self.sample_normal)

        [R13, R23, R33] = np.dot(self.rot_matrix, np.asarray([0., 0., 1.]))

        dr = self.ratioDw*self.ImageWidth

        # pixel position in lab referece frame, both centered on detector
        # this is code from RQconv.c
        # In comparison, the coordinates in equ(18) in the reference above are centered at the sample
        tt = (R13*np.cos(Phi)+R23*np.sin(Phi))*np.tan(2.0*Theta)
        Z1 = dr*tt/(tt-R33);
        X1 = (dr-Z1)*np.tan(2.0*Theta)*np.cos(Phi);
        Y1 = (dr-Z1)*np.tan(2.0*Theta)*np.sin(Phi);
        
        R1 = np.vstack((X1, Y1, Z1))
        
        # transform to detector frame
        [X, Y, Z] = np.dot(self.rot_matrix.T, R1);

        # pixel position, note reversed y-axis
        X0[idx] = X + self.bm_ctr_x
        Y0[idx] = -Y + self.bm_ctr_y
        X0[~idx] = np.nan
        Y0[~idx] = np.nan
        
        return (X0, Y0)
    
class ExpParaLiX(ExpPara):
    """
    This is one way to define the orientation of the detector, as defined in the ref above
    The detector orientation can be defined by a different set of angles
    A different derived class can be defined to inherit the same fuctions from ExpPara 
    """
    det_orient = 0.
    det_tilt = 0.
    det_phi = 0.
    
    def calc_rot_matrix(self):
        tm1 = RotationMatrix('z', np.radians(-self.det_orient))
        tm2 = RotationMatrix('y', np.radians(self.det_tilt))
        tm3 = RotationMatrix('z', np.radians(self.det_orient+self.det_phi))
        self.rot_matrix = np.dot(np.dot(tm3, tm2), tm1)

    
class MatrixWithCoords:
    # 2D data with coordinates
    d = None
    xc = None
    yc = None 
    datatype = None
    
    def copy(self):
        ret = MatrixWithCoords()
        ret.xc = self.xc
        ret.yc = self.yc
        ret.d = np.copy(self.d)
        
        return ret
    
    def conv(self, Nx1, Ny1, xc1, yc1, mask=None, cor_factor=1, datatype=DataType.det):
        """ Nx1, Ny1 can be either the number of bins or an array that specifies the bin edges
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

    def roi(self, x1, x2, y1, y2, mask=None):
        ret = MatrixWithCoords()      
        ret.datatype = self.datatype
        
        xidx = (self.xc>=np.min([x1,x2])) & (self.xc<=np.max([x1,x2]))
        yidx = (self.yc>=np.min([y1,y2])) & (self.yc<=np.max([y1,y2]))
        t1 = np.tile(xidx, [len(yidx),1])
        t2 = np.tile(np.flipud(yidx), [len(xidx),1]).T

        ret.xc = self.xc[xidx]
        ret.yc = self.yc[yidx]
        ret.d = np.asarray(self.d[t1*t2].reshape((len(ret.yc),len(ret.xc))), dtype=float)
        
        if mask is not None:
            idx = mask.map[t1*t2].reshape((len(ret.yc),len(ret.xc)))
            ret.d[idx] = np.nan
        
        return ret

    # return the averaged value of the data at coordinates (x0,y0), with a box of (dx, dy) 
    def val(self, x0, y0, dx, dy, mask=None):
        roi = self.roi(x0-0.5*dx, x0+0.5*dx, y0-0.5*dy, y0+0.5*dy, mask=mask)
        d = roi[~np.isnan(roi)]
        return np.sum(d)/len(d)
    
    # unit of ang is degrees
    def line_cut(self, x0, y0, ang, lT, lN, nT, nN, mask=None):
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
    
    # collapse the matrix into an array along x coordinates if axis=0, along y coordinates if axis=1
    def flatten(self, axis=0):
        cor = np.ones(self.d.shape)
        data = np.copy(self.d)
        
        idx = np.isnan(data)
        cor[idx] = 0
        data[idx] = 0
        
        if axis!=0 and axis!=1:
            raise Exception("unkown axis for flattening data.")
        
        # exclude blank portion of the data
        dcor = np.sum(cor, axis=axis)
        dd = np.sum(data, axis=axis)
        idx = (dcor<=0)
        dcor[idx] = 1.
        dd[idx] =np.nan
        dd /= dcor

        if axis==0:
            return dd 
        elif axis==1:
            # the data is always stored with the upper left corner first in the memory
            # the index is stored with 
            return np.flipud(dd)

    # average with the list of data given, all data must have the same coordinates and datatype
    def average(self, datalist):
        ret = copy.deepcopy(self)
        for d in datalist:
            if not np.array_equal(ret.xc, d.xc) or not np.array_equal(ret.xc, d.xc) or ret.datatype!=d.datatype:
                raise Exception("attempted average between imcompatiple data.")
            ret.d += d.d
        ret.d /= (len(datalist)+1)
        
        return ret
    
    # merge with the list of data given, not necessarily having the same coordinates
    def merge(self, datalist):
        pass 


class Data2d:
    """ 2D scattering data class
    stores the scattering pattern itself, 
    """

    def __init__(self, filename, im=None, timestamp=None, uid='', exp=None):
        """ read 2D scattering pattern
        will rely on Fabio to recognize the file format 
        """
        self.exp = None
        self.timestamp = None
        self.uid = None
        self.data = MatrixWithCoords()
        self.qrqz_data = MatrixWithCoords()
        self.qphi_data = MatrixWithCoords()
        
        if im is None:
            f = fabio.open(filename)
            self.im = f.data
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
        else:
            self.im = im
            self.timestamp = timestamp
            self.uid = uid

        # self.im always stores the original image
        # self.data store the array data after the flip operation
        if exp is not None:
            self.set_exp_para(exp)
        else:
            # temporarily set data to img, without a defined exp_para
            self.flip(0)
            self.height, self.width = self.data.d.shape

    def set_timestamp(self, ts):
        # ts must be a valid datetime structure
        self.timestamp = ts

    def flip(self, flip):
        """ this is a little complicated
            if flip<0, do a mirror operation first
            the absolute value of flip is the number of 90-deg rotations
        """  
        self.data.d = np.asarray(self.im).copy()
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
        
    def conv_Iq(self, qgrid, mask=None, cor_factor=1):
        
        dd = self.data.d/cor_factor
        
        # Pilatus might use negative values to mark deak pixels
        idx = (self.data.d>=0)
        if mask is not None:
            idx &= ~(mask.map)
            
        qd = self.exp.Q[idx].flatten()
        dd = np.asarray(dd[idx].flatten(), dtype=np.float)  # other wise dd*dd might become negative

        # generate bins from qgrid, 
        bins = np.append([2*qgrid[0]-qgrid[1]], qgrid) 
        bins += np.append(qgrid , [2*qgrid[-1]-qgrid[-2]])
        bins *= 0.5

        norm,edges = np.histogram(qd, bins=bins, weights=np.ones(len(qd))) 
        Iq,edges = np.histogram(qd, bins=bins, weights=dd)
        Iq2,edges = np.histogram(qd, bins=bins, weights=dd*dd)

        Iq[norm>0] /= norm[norm>0]
        Iq2[norm>0] /= norm[norm>0]
        dI = np.sqrt(Iq2-Iq*Iq)
        dI[norm>0] /= np.sqrt(norm[norm>0])
        Iq[norm<=0] = np.nan
        dI[norm<=0] = np.nan
        
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
        toolbar = plt.get_current_fig_manager().toolbar
        x = event.xdata 
        y = event.ydata 
        if self.coordinate_translation=="xy2qrqz":
            (q, phi, qr, qz) = self.exp.calc_from_XY(np.asarray([x]),np.asarray([y]))
            msg = "qr=%.4f , qz=%.4f" % (qr[0], qz[0])
        elif self.coordinate_translation=="xy2qphi":
            (q, phi, qr, qz) = self.exp.calc_from_XY(np.asarray([x]),np.asarray([y]))
            msg = "q=%.4f, phi=%.1f" % (q[0],phi[0])
        else:
            msg = ""
        toolbar.set_message(msg)
        return True

    # best modify d2.xc/d2.yc instead of using xscale/yscale
    def plot(self, mask=None, log=False, aspect=1., xscale=1., yscale=1.):

        dd = np.asarray(self.d2.d, dtype=np.float)
        if mask is not None:
            dd[mask.map] = np.nan

        immax = np.average(dd) + 5 * np.std(dd)
        immin = np.average(dd) - 5 * np.std(dd)
        if immin < 0:
            immin = 0

        if log:
            self.img = self.ax.imshow(dd, aspect=aspect,
                                      cmap=self.cmap, interpolation='nearest', norm=LogNorm(),
                                      extent=[self.d2.xc[0]*xscale, self.d2.xc[-1]*xscale, 
                                              self.d2.yc[0]*yscale, self.d2.yc[-1]*yscale])
        else:
            self.img = self.ax.imshow(dd, vmax=immax, vmin=immin, aspect=aspect,
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
            sucrose: 
            CeO2:
            LaB6: 
        """
        if std=="AgBH":
            q_std = np.hstack((0.1076*np.arange(1,10), [1.37]))
        elif std=="sucrose":
            q_std = [0.5933, 0.8289, 0.9054, 0.9336, 1.1000]
        elif std=="CeO2": # http://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nbsmonograph25-20.pdf
            q_std = [ 2.0116, 1.8985, 3.2838, 3.8504 ]
        elif std=="LaB6": # NIST SRM 660c
            q_std = [ 1.5115, 2.1376, 2.6180, 3.0230, 3.3799, 3.7025 ]
        else: # unknown standard
            return 
        self.mark_coords(q_std, [], sym, DataType.q)
    
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
        

