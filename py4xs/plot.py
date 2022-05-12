import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm

from py4xs.data2d import DataType,Data2d
from py4xs.utils import auto_clim

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
             aspect='auto', xscale=1., yscale=1.):

        dd = np.asarray(self.d2.d, dtype=float)

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
            q_std = np.asarray(std, dtype=float)
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
        

def show_data(d2s, detectors=None, ax=None, fig=None, showRef=None,
              logScale=True, showMask=True, mask_alpha=0.1, 
              aspect='auto', clim='auto', cmap=None, **kwargs):
    """ d2s should be a dictionary
    """
    
    if isinstance(d2s, dict):
        ndet = len(d2s)
        if ndet>0 and fig is None:
            fig = plt.figure()
        i = 0
        for ext,d2 in d2s.items():
            i += 1
            fig.add_subplot(1, ndet, i)
            ax = plt.gca()
            show_data(d2, ax=ax, aspect=aspect,
                      logScale=logScale, showMask=showMask, mask_alpha=mask_alpha, 
                      clim=clim, showRef=showRef, cmap=cmap, **kwargs)
    elif isinstance(d2s, Data2d):
        d2 = d2s
        if ax is None:
            plt.figure()
            ax = plt.gca()
        pax = Axes2dPlot(ax, d2.data, exp=d2.exp)
        if cmap is not None:
            pax.set_color_scale(plt.get_cmap(cmap)) 
        pax.plot(logScale=logScale, showMask=showMask, mask_alpha=mask_alpha, aspect=aspect)
        if clim=="auto":
            clim = auto_clim(d2.data.d[~d2.exp.mask.map], logScale)
        pax.img.set_clim(*clim)
        pax.coordinate_translation="xy2qphi"
        if showRef:
            pax.mark_standard(*showRef)
        ax.set_title(f"frame #{d2.md['frame #']}")
        pax.capture_mouse()


def show_data_qxy(d2s, detectors, ax=None, dq=0.01, bkg=None,
                  logScale=True, useMask=True, clim=(0.1,14000), 
                  aspect=1, cmap=None, colorbar=False, **kwargs):
    
    if ax is None:
        plt.figure()
        ax = plt.gca()

    xqmax = np.max([d.exp_para.xQ.max() for d in detectors])
    xqmin = np.min([d.exp_para.xQ.min() for d in detectors])
    yqmax = np.max([d.exp_para.yQ.max() for d in detectors])
    yqmin = np.min([d.exp_para.yQ.min() for d in detectors])

    xqmax = np.floor(xqmax/dq)*dq
    xqmin = np.ceil(xqmin/dq)*dq
    yqmax = np.floor(yqmax/dq)*dq
    yqmin = np.ceil(yqmin/dq)*dq

    xqgrid = np.arange(start=xqmin, stop=xqmax+dq, step=dq)
    yqgrid = np.arange(start=yqmin, stop=yqmax+dq, step=dq)        

    xyqmaps = []
    for det in detectors:
        dn = det.extension
        if useMask:
            mask = d2s[dn].exp.mask
        else:
            mask = None

        cor_factor = d2s[dn].exp.FSA*d2s[dn].exp.FPol
        if det.flat is not None:
            cor_factor = det.flat*cor_factor
        dm = d2s[dn].data.conv(xqgrid, yqgrid, d2s[dn].exp.xQ, d2s[dn].exp.yQ, 
                               mask=mask, cor_factor=cor_factor)

        if bkg is not None:
            if dn in bkg.keys():
                dbkg = Data2d(bkg[dn], exp=det.exp_para)
                dm_b = dbkg.data.conv(xqgrid, yqgrid, d2s[dn].exp.xQ, d2s[dn].exp.yQ, 
                                      mask=mask, cor_factor=cor_factor)
                dm.d -= dm_b.d    

        dm.d *= (d2s[dn].exp.Dd/d2s["_SAXS"].exp.Dd)**2
        xyqmaps.append(dm)

    dm = xyqmaps[0].merge(xyqmaps[1:])
    dm.xc = xqgrid
    dm.xc_label = "qx"
    dm.xc_prec = 3
    dm.yc = yqgrid
    dm.yc_label = "qy"
    dm.yc_prec = 3

    if clim=="auto":
        clim = auto_clim(dm.d, logScale)
    dm.plot(ax=ax, logScale=logScale, clim=clim, aspect=aspect, colorbar=colorbar)
    ax.set_title(f"frame #{d2s[list(d2s.keys())[0]].md['frame #']}")
        
    
def show_data_qphi(d2s, detectors, ax=None, Nq=200, Nphi=60,
                   apply_symmetry=False, fill_gap=False, sc_factor=None, 
                   logScale=True, useMask=True, clim=(0.1,14000), bkg=None,
                   aspect="auto", cmap=None, colorbar=False, **kwargs):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if isinstance(Nq, int):
        qmax = np.max([d.exp_para.Q.max() for d in detectors]) 
        qmin = np.min([d.exp_para.Q.min() for d in detectors]) 
        # keep 2 significant digits only for the step_size
        dq = (qmax-qmin)/Nq
        n = int(np.floor(np.log10(dq)))
        sc = np.power(10., n)
        dq = np.around(dq/sc, 1)*sc

        qmax = dq*np.ceil(qmax/dq)
        qmin = dq*np.floor(qmin/dq)
        Nq = int((qmax-qmin)/dq+1)

        q_grid = np.linspace(qmin, qmax, Nq) 
    else:
        q_grid=Nq
        qmin = q_grid[0]
        qmax = q_grid[-1]

    Nphi = 2*int(Nphi/2)+1
    phi_grid = np.linspace(-180., 180, Nphi)

    dms = []
    for det in detectors:
        dn = det.extension
        if not dn in d2s.keys():
            continue
        if useMask:
            mask = d2s[dn].exp.mask
        else:
            mask = None

        cor_factor = d2s[dn].exp.FSA*d2s[dn].exp.FPol
        if det.flat is not None:
            cor_factor = det.flat*cor_factor

        # since the q/phi grids are specified, use the MatrixWithCoord.conv() function
        # the grid specifies edges of the bin for histogramming
        # the phi value in exp_para may not always be in the range of (-180, 180)
        dm = d2s[dn].data.conv(q_grid, phi_grid, d2s[dn].exp.Q, d2s[dn].exp.Phi,
                               #fix_angular_range(d2s[dn].exp.Phi),
                               cor_factor=cor_factor, 
                               mask=mask, datatype=DataType.qphi)

        if bkg is not None:
            if dn in bkg.keys():
                dbkg = Data2d(bkg[dn], exp=det.exp_para)
                dm_b = dbkg.data.conv(q_grid, phi_grid, d2s[dn].exp.Q,d2s[dn].exp.Q, d2s[dn].exp.Phi,
                                      #fix_angular_range(d2s[dn].exp.Phi),
                                      cor_factor=cor_factor,
                                      mask=mask, datatype=DataType.qphi)
                dm.d -= dm_b.d    

        dm.d /= det.fix_scale

        if apply_symmetry:
            dm = dm.apply_symmetry()
        if fill_gap:
            dm = dm.fill_gap(method='linear')

        dms.append(dm)

    dm = dms[0].merge(dms[1:])
    dm.xc = q_grid
    dm.xc_label = "q"
    dm.xc_prec = 3
    dm.yc = phi_grid
    dm.yc_label = "phi"
    dm.yc_prec = 1

    dm.plot(ax=ax, sc_factor=sc_factor, logScale=logScale, clim=clim, cmap=cmap, colorbar=colorbar)
    ax.set_title(f"frame #{d2s[list(d2s.keys())[0]].md['frame #']}")

    return dm