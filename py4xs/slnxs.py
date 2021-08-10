import numpy as np
import matplotlib.pyplot as plt
import collections

from py4xs.data2d import Data2d
from py4xs.mask import Mask
from py4xs.utils import common_name,smooth

import os
import sys
import time
import copy
import itertools as it
import multiprocessing as mp

# this is the ratio between protein average denstiy and water density
# it is assumed to be a constant but in reality depends on the specific portein
# see Fischer et.al.  Protein Sci. 2004 October; 13(10): 2825-2828
PROTEIN_WATER_DENSITY_RATIO = 1.35

# Each Data1d corresponds to one single scattering pattern
# The intensity is normalized based on
#     (1) beam intensity through the beam stop, as in prvious version
#  or (2) WAXS intensity (water scattering)
#  or (3) an externally dtermined value
# The intensity can be further normalized a reference trans value, so
#   that different sets can be compared.
# Data1d sets (must share the same qgrid) can be merged: e.g. SAXS and WAXS
# Background subtraction and flat field correction are also supported

TRANS_EXTERNAL = 0
TRANS_FROM_BEAM_CENTER = 1
TRANS_FROM_WAXS = 2

from enum import Enum
# removed from_beam_center since it is really external
# also it is difficult to keep track when merging multiple detectors
class trans_mode(Enum):
    external = 0
    from_waxs = 2

# trans_mode=TRANS_FROM_BEAM_CENTER
# this works if there is a semi-transparent beam stop
BEAM_SIZE_hW = 5
BEAM_SIZE_hH = 4

# this is the global setting
TRANS_MODE = trans_mode.from_waxs
# this is the minimum intensity to be used for trans calculations
WAXS_THRESH = 10


# this is the scaling factor for indivudual curves that belong to the same sample
# they are offset for clarity in the plots
VOFFSET = 1.5

font_size_list = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
def get_font_size(size_index):
    """ "medium" has size_index of 0
        the size_index is negative for smaller fonts and possitive for larger ones  
    """
    if size_index in font_size_list:
        i = font_size_list.index(size_index)
    else:
        i = int(size_index)+3
        if i<0:
            i = 0
        elif i>=len(font_size_list):
            i = len(font_size_list)-1
    return i-3,font_size_list[i]
        
class Data1d:
    def __init__(self, transMode=None):
        self.comments = ""
        self.label = "data"
        self.overlaps = []
        self.raw_data = {}
        self.timestamp = None
        self.trans_w = -1
        self.trans_e = -1
        self.trans = -1
        self.transMode = transMode
        
    def load_from_2D(self, image, exp_para, qgrid, pre_process=None, flat_cor=None,
                     mask=None, save_ave=False, debug=False, label=None, dtype=None):
        """
        image: a filename, or a Data2d instance, or a numpy array
        qgrid: for the 1D data
        exp_para: ExpPara
        mask: no longer used, extract from exp_para
        """
        self.qgrid = qgrid
        mask = exp_para.mask

        if debug==True:
            print("loading data from 2D image: ", label)
    
        if isinstance(image, Data2d):
            d2 = image
        else:
            d2 = Data2d(image, exp=exp_para, dtype=dtype)
            self.timestamp = d2.timestamp
            self.label = d2.label
            self.timestamp = d2.timestamp
            
        if label is not None:
            self.label = label
            
        # place holder for pre-processing, to deal with things like 
        # dark current, flat field, and dezinger corrections on the 2D data
        if pre_process is not None:
            pre_process(d2.data)
        
        cor_factor = exp_para.FSA*exp_para.FPol
        if flat_cor is not None:
            cor_factor *= flat_cor    # for rescuing data with incorrect flat field at the time of collection
        self.data,self.err = d2.conv_Iq(qgrid, mask, cor_factor = cor_factor)  

        if isinstance(image, np.ndarray):
            del d2      # d2 is only used temporarily
        
        if save_ave and isinstance(image, str):
            self.save(image + ".ave", debug=debug)     
        

    def set_trans(self, transMode=None, trans=-1, ref_trans=-1,
                  calc_water_peak=False, q_start=1.85, q_end=2.15, debug=False):
        """
        normalize intensity, from trans to ref_trans
        trans can be either from the beam center/beam stop or water scattering
        sometimes the data may require both measures of transmitted intensity
        
        this operation should be performed after SAXS/WAXS merge, because
          1. SAXS and WAXS should have the same trans
          2. if trans_mode is TRNAS_FROM_WAXS, the trans value needs to be calculated from WAXS data
        
        """
        if transMode is None and self.transMode is not None:
            transMode = self.transMode
        else:
            assert(isinstance(transMode, trans_mode))
            self.transMode = transMode
        if self.transMode==trans_mode.from_waxs or calc_water_peak:
            # get trans for the near the maximum in the WAXS data
            # for solution scattering, hopefully this reflect the intensity of water scattering
            idx = (self.qgrid > q_start) & (self.qgrid < q_end)  # & (self.data>0.5*np.max(self.data))
            if len(self.qgrid[idx]) < 5:
                print("not enough data points under the water peak, consider using a different trans_mode.")
                #raise Exception()
            
            # trying to narrow down the peak range turns out to be a bad idea
            # the width then could vary between datasets, creating artificial fluctuation in trans 
            #idx1 = idx & (self.data >= 0.95*np.max(self.data[idx]))

            if (self.data[idx]<WAXS_THRESH).all() and debug!='quiet':
                print("the data points for trans calculation are below WAXS_THRESH: ", 
                      np.max(self.data[idx]), WAXS_THRESH)                
            self.trans_w = np.sum(self.data[idx])
            if self.transMode==trans_mode.from_waxs:
                self.trans = self.trans_w 
            qavg = np.average(self.qgrid[idx])
            if self.trans_w<1.0:
                print(f'caluclated trans is {self.trans_w}, setting it artifically to WAXS_THRESH.')
                self.trans_w = WAXS_THRESH
            if debug==True:
                print("using data near the high q end (q~%f)" % qavg, end=' ')
            self.comments += "# transmitted beam intensity from WAXS (q~%.2f)" % qavg
        if self.transMode==trans_mode.external and trans>=0: 
            # if the trans value is specified, by definition transMode should be external
            # if trans=-1, it is not meant to be the value to be set
            #if trans<0:
            #    print(f"Warning: {trans} is not a valid value for transmitted intensity.")
            #    trans = 0
            self.comments += f"# transmitted beam intensity given externally: {trans}"
            self.trans_e = trans
            self.trans = trans
            #if self.transMode==trans_mode.external:
            #    self.trans = trans

        self.comments += ": %f \n" % self.trans
        if debug==True:
            print("trans for %s set to %f" % (self.label, self.trans))

        if ref_trans > 0:
            if self.trans<0:
                print(f"cannot normalize intensity since data1d does not have a valid trans value: {self.trans}")
            self.comments += "# scattering intensity normalized to ref_trans = %f \n" % ref_trans
            self.data *= ref_trans/self.trans
            self.err *= ref_trans/self.trans
            for ov in self.overlaps:
                ov['raw_data1'] *= ref_trans/self.trans
                ov['raw_data2'] *= ref_trans/self.trans
            self.trans_w *= ref_trans/self.trans
            self.trans_e *= ref_trans/self.trans
            self.trans = ref_trans
            if debug==True:
                print("normalized to %f" % ref_trans)


    def avg(self, dsets, weighted=False, qmax_for_weight=0.3, 
            plot_data=False, ax=None, debug=False, fontsize='large'):
        """
        dsets is a collection of Data1d
        weighted:
            if False 
                the dsets are simply averaged together 
                errorbar is increased if there are discrepencies between individual values?? 
            otherwise 
                weight should contain a list of weight factors, corresponding to each dset
                each dset is first scaled by the 
                a weighted average (smaller errorbar has higher weight) is then performed
        ax is the Axes to plot the data in
        TODO: should calculate something like the cross-correlation between sets
        to evaluate the consistency between them
        """
        if debug!='quiet':
            print("averaging data with %s: \n" % self.label, end=' ')
        i_fs = get_font_size(fontsize)[0]
        
        n = 1
        if plot_data:
            if ax is None:
                plt.figure()
                plt.subplots_adjust(bottom=0.15)
                ax = plt.gca()
            ax.set_xlabel("$q (\AA^{-1})$", fontsize=get_font_size(i_fs)[1])
            ax.set_ylabel("$I$", fontsize=get_font_size(i_fs)[1])
            ax.set_xscale('log')
            ax.set_yscale('log')
            idx = (self.data > 0)
            ax.errorbar(self.qgrid[idx], self.data[idx], self.err[idx], label=self.label)
            for ov in self.overlaps:
                ax.plot(ov['q_overlap'], ov['raw_data1'], "v")
                ax.plot(ov['q_overlap'], ov['raw_data2'], "^")
        
        d0 = copy.deepcopy(self)
        if len(dsets)==0:
            return d0

        if weighted:
            wt = []
            for d in dsets+[self]:
                w0 = np.sum(np.fabs(d.data[d.qgrid<qmax_for_weight]))
                if w0<=0:
                    raise Exception(f"weight for averaging <0: {w0}")
                wt.append(w0)
            wt = np.asarray(wt)
            wt /= wt.max()
            er2 = (self.err/wt[-1])**2
            d0.err = 1./er2
            d0.data = d0.data/er2 
            if debug==True:
                print("weight factors: ", wt)
        
        for i in range(len(dsets)):
            d1 = dsets[i]
            if debug==True:
                print("%s " % d1.label, end=' ')
            if not (d0.qgrid == d1.qgrid).all():
                raise Exception("\n1D sets cannot be averaged: qgrid mismatch")

            d0.trans += d1.trans
            d0.trans_w += d1.trans_w
            d0.trans_e += d1.trans_e
            if weighted:
                er2 = (d1.err/wt[i])**2
                d0.err += 1/er2
                d0.data += d1.data/er2 
            else:
                d0.data += d1.data
                d0.err += d1.err

            #if self.transMode == trans_mode.from_beam_center:
            #    d0.roi += d1.roi
            d0.comments += "# averaged with \n%s" % d1.comments.replace("# ", "## ")
            if plot_data:
                idx = (d1.data > 0)  # Remove Zeros on plot
                ax.errorbar(d1.qgrid[idx], d1.data[idx] * VOFFSET ** n, d1.err[idx] * VOFFSET ** n, label=d1.label)
            for i in range(len(d1.overlaps)):
                if plot_data:
                    ax.plot(d1.overlaps[i]['q_overlap'], d1.overlaps[i]['raw_data1'] * VOFFSET ** n, "v")
                    ax.plot(d1.overlaps[i]['q_overlap'], d1.overlaps[i]['raw_data2'] * VOFFSET ** n, "^")
                d0.overlaps[i]['raw_data1'] += d1.overlaps[i]['raw_data1']
                d0.overlaps[i]['raw_data2'] += d1.overlaps[i]['raw_data2']
            n += 1
            d0.label = common_name(d0.label, d1.label)

        d0.trans /= n
        d0.trans_w /= n
        d0.trans_e /= n
        if weighted:
            d0.data /= d0.err
            d0.err = 1./np.sqrt(d0.err)
        else:    
            d0.data /= n
            d0.err /= (n*np.sqrt(n))   # should not be just sqrt(n), that would increase err after averaging
        #if self.transMode == trans_mode.from_beam_center:
        #    d0.roi /= n
        for ov in d0.overlaps:
            ov['raw_data1'] /= n
            ov['raw_data2'] /= n
        if debug==True:
            print("\naveraged set re-named to %s." % d0.label)

        if plot_data:
            # plot the averaged data over each individual curve
            for i in range(n):
                if i == 0:
                    idx = (d0.data > 0)  # Remove Zeros on plot
                    handles, labels = ax.get_legend_handles_labels()
                    lbl = "averaged" if "averaged" not in labels else ""
                    ax.plot(d0.qgrid[idx], d0.data[idx] * VOFFSET ** i, color="gray", lw=2, ls="--", label=lbl)
                else:
                    idx = (d0.data > 0)  # Remove Zeros on plot
                    ax.plot(d0.qgrid[idx], d0.data[idx] * VOFFSET ** i, color="gray", lw=2, ls="--")
            leg = ax.legend(loc='upper right', frameon=False)

            for t in leg.get_texts():
                t.set_fontsize(get_font_size(i_fs-2)[1])

        return d0


    def bkg_cor(self, dbak, sc_factor=1., plot_data=False, ax=None, 
                inplace=False, check_overlap=False, show_eb=True, debug=False, fontsize='large'):
        """
        background subtraction
        """
        dset = None
        if inplace:
            dset = self
        else:
            dset = copy.deepcopy(self)
        i_fs = get_font_size(fontsize)[0]
            
        if debug==True:
            print("background subtraction: %s - %s" % (dset.label, dbak.label))
        if not (dbak.qgrid == dset.qgrid).all():
            print("background subtraction failed: qgrid mismatch")
            sys.exit()
        if dset.trans < 0 or dbak.trans <= 0:
            print("WARNING: trans value not assigned to data or background, assuming normalized intensity.")
            sc = 1.
        else:
            sc = dset.trans / dbak.trans

        # need to include raw data
        if plot_data:
            if ax is None:
                plt.figure()
                plt.subplots_adjust(bottom=0.15)
                ax = plt.gca()
            ax.set_xlabel("$q (\AA^{-1})$", fontsize=get_font_size(i_fs)[1])
            ax.set_ylabel("$I$", fontsize=get_font_size(i_fs)[1])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_tick_params(labelsize=get_font_size(i_fs-1)[1])
            ax.yaxis.set_tick_params(labelsize=get_font_size(i_fs-1)[1])
            idx = (dset.data > 0) & (dbak.data > 0)
            ax.plot(dset.qgrid[idx], dset.data[idx], label=self.label)
            ax.plot(dbak.qgrid[idx], dbak.data[idx], label=dbak.label)
            ax.plot(dbak.qgrid[idx], dbak.data[idx] * sc * sc_factor, label=dbak.label + ", scaled")

        if len(dset.overlaps) != len(dbak.overlaps):
            if check_overlap:
                raise Exception("Background subtraction failed: overlaps mismatch.")
        else:
            for i in range(len(dset.overlaps)):
                dset.overlaps[i]['raw_data1'] -= dbak.overlaps[i]['raw_data1'] * sc_factor * sc
                dset.overlaps[i]['raw_data2'] -= dbak.overlaps[i]['raw_data2'] * sc_factor * sc
                if plot_data:
                    ax.plot(dset.overlaps[i]['q_overlap'], dset.overlaps[i]['raw_data1'], "v")
                    ax.plot(dset.overlaps[i]['q_overlap'], dset.overlaps[i]['raw_data2'], "^")
        if plot_data:
            leg = ax.legend(loc='upper right', frameon=False)
            for t in leg.get_texts():
                t.set_fontsize(get_font_size(i_fs-2)[1])

        if debug==True:
            print("using scaling factor of %f" % (sc * sc_factor))
        dset.data -= dbak.data * sc * sc_factor
        dset.err += dbak.err * sc * sc_factor
        if plot_data:
            if show_eb:
                ax.errorbar(dset.qgrid, dset.data, dset.err)
            else:
                ax.plot(dset.qgrid, dset.data)
                
        dset.comments += "# background subtraction using the following set, scaled by %f (trans):\n" % sc
        if not sc_factor == 1.:
            dset.comments += "# with addtional scaling factor of: %f\n" % sc_factor
        dset.comments += dbak.comments.replace("# ", "## ")

        return dset

    def scale(self, sc):
        """
        scale the data by factor sc
        """
        if sc <= 0:
            print("scaling factor is non-positive: %f" % sc)
        self.data *= sc
        self.err *= sc
        self.trans *= sc
        if self.trans_w>0:
            self.trans_w *= sc
        if self.trans_e>0:
            self.trans_e *= sc
        self.comments += "# data is scaled by %f.\n" % sc
        if len(self.overlaps) != 0:
            for ov in self.overlaps:
                ov['raw_data1'] *= sc
                ov['raw_data2'] *= sc
                
        return self

    def merge(self, d1, qmax=-1, qmin=-1, fix_scale=-1, debug=False):
        """
        combine the data in self and d1
        scale d1 intensity to match self
        self and d1 should have the same qgrid

        if qmax or qmin <0
        simply keep the WAXS data that is beyond qmax for the SAXS data
        this is useful for utilizing WAXS to normalize intensity but keep SAXS data only
        """

        if debug==True:
            print("merging data: %s and %s ..." % (self.label, d1.label))
        if not (d1.qgrid == self.qgrid).all():
            print("merging data sets should have the same qgrid.")
            exit()

        # this gives the overlapping region
        idx = (self.data > 0) & (d1.data > 0)

        if len(self.qgrid[idx]) > 0:
            qmin0 = min(d1.qgrid[idx])
            qmax0 = max(self.qgrid[idx])
            # merge SAXS/WAXS based on intensity in the overlapping region
            if qmax0 < qmax:
                qmax = qmax0
            if qmin0 > qmin:
                qmin = qmin0
            idx = (self.qgrid > qmin) & (self.qgrid < qmax)
            # save the raw data in case needed, e.g. for ploting
            self.overlaps.append({'q_overlap': self.qgrid[idx],
                                  'raw_data1': self.data[idx],
                                  'raw_data2': d1.data[idx]})
        else:
            # no overlap
            # simply stack WAXS data to the high q end of SAXS data
            qmin = qmax = max(self.qgrid[self.data > 0])
            self.overlaps.append({'q_overlap': np.empty(0),
                                  'raw_data1': np.empty(0),
                                  'raw_data2': np.empty(0)})

        # idx = np.asarray([],dtype=int)

        if len(self.qgrid[idx])==0:
            if debug!='quiet':
                print("data sets are not overlapping in the given q range.")
            if fix_scale < 0:
                fix_scale = 1
                if debug!='quiet':
                    print("forcing fix_scale=1.")
        elif len(self.qgrid[idx]) < 5 and debug!='quiet':
            print("too few overlapping points: %d" % len(self.qgrid[idx]))

        if fix_scale > 0:
            # For a given experimental configuration, the intensity normlization
            # factor between the SAXS and WAXS should be well-defined. This factor
            # can be determined using scattering data with siginificant intensity
            # in the overlapping q-range and applied to all data collected in the
            # same configuration.
            sc = fix_scale
        else:
            sc = np.linalg.lstsq(np.asmatrix(self.data[idx]).T, np.asmatrix(d1.data[idx]).T)[0]
            sc = np.trace(sc)

        d1.data /= sc
        d1.err /= sc
        if len(self.qgrid[idx]) > 0:
            if debug==True:
                print("Scaled Overlaps by 1/%f" % sc)
            self.overlaps[-1]['raw_data2'] /= sc
            self.overlaps[-1]['sc'] = sc

        self.label = common_name(self.label, d1.label)
        if debug==True:
            print("set2 scaled by 1/%f" % sc)
            print("merged set re-named %s." % self.label)

        if len(self.qgrid[idx]) > 0:
            # averaging using 1/err^2 as weight, maximum likelihood estimator 
            w1 = 1./(self.err[idx]*self.err[idx]) 
            w2 = 1./(d1.err[idx]*d1.err[idx]) 
            self.data[idx] = (w1*self.data[idx] + w2*d1.data[idx]) / (w1+w2)
            self.err[idx] = np.sqrt((w1/(w1+w2)*self.err[idx])**2 + (w2/(w1+w2)*d1.err[idx])**2)
        self.data[self.qgrid >= qmax] = d1.data[self.qgrid >= qmax]
        self.err[self.qgrid >= qmax] = d1.err[self.qgrid >= qmax]

        self.comments += "# merged with the following set by matching intensity within (%.4f, %.4f)," % (qmin, qmax)
        self.comments += " scaled by %f\n" % sc
        self.comments += d1.comments.replace("# ", "## ")

    def plot_Guinier(self, qs=0, qe=10, rg=15, fix_qe=False, scale_wabs=-1,
                     ax=None, no_plot=False, fontsize="large"):
        """ do Gunier plot, estimate Rg automatically
        qs specify the lower end of the q-range to perform the fit in
        rg is the optinal initial estimate
        if fix_qe==1, qe defined the end of the region to perform the fit
        """
        idx = (self.data > 0)
        i_fs = get_font_size(fontsize)[0]
        # print self.data

        scale = 1.0
        if scale_wabs>0:
            assert(self.trans_w>0)
            scale *= scale_wabs/self.trans_w

        if no_plot==False:
            if ax is None:
                ax = plt.gca()
            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.errorbar(self.qgrid[idx]**2, self.data[idx]*scale, self.err[idx]*scale)

        cnt = 0
        t = self.qgrid[self.data > 0][0]
        if qs < t: qs = t
        while cnt < 10:
            if (not fix_qe) and qe > 1.3/rg and 1.3/rg > qs+0.004: qe = 1.3/rg
            td = np.vstack((self.qgrid, self.data))
            td = td[:, td[0, :] >= qs]
            td = td[:, td[0, :] <= qe]
            td[0, :] = td[0, :] * td[0, :]
            td[1, :] = np.log(td[1, :])
            rg, i0 = np.polyfit(td[0, :], td[1, :], 1)
            i0 = np.exp(i0)
            if rg<0:
                rg = np.sqrt(-rg * 3.)
            else:
                rg = 1e-6   # 
                print("likely strong inter-particle interaction ...")
                break
            cnt += 1
            # print i0, rg
        td[1, :] = i0 * np.exp(-td[0, :]*rg*rg/3.)
        n1 = len(self.qgrid[self.qgrid<qs])
        n2 = len(self.qgrid)-len(self.qgrid[self.qgrid>qe])
        fit_range = [n1,n2]
        i0*=scale
        
        if no_plot==False and rg>0.1:
            #ax.tick_params(axis='y', labelleft=False)    
            ax.plot([td[0, 0], td[0, -1]], [td[1, 0]*scale, td[1, -1]*scale], "ro")
            ax.plot(self.qgrid**2, i0*np.exp(-(self.qgrid*rg)**2/3))
            ax.set_ylabel("$I$", fontsize=get_font_size(i_fs)[1])
            ax.set_xlabel("$q^2 (\AA^{-2})$", fontsize=get_font_size(i_fs)[1])
            ax.xaxis.set_tick_params(labelsize=get_font_size(i_fs-1)[1])
            ax.yaxis.set_tick_params(labelsize=get_font_size(i_fs-1)[1])
            # plt.subplots_adjust(bottom=0.15)
            ax.set_xlim(0, qe**2*1.2)
            #ax.autoscale_view(tight=True, scalex=False, scaley=True)
            if i0>0:
                ax.set_ylim(top=i0*2, bottom=i0*np.exp(-(qe*rg)**2/3)/2)
        # print "I0=%f, Rg=%f" % (i0,rg)
        return (i0, rg, fit_range) # include fit range to be more compatible with ATSAS

    def plot_pr(self, i0, rg, qmax=5., dmax=200., ax=None, fontsize='large'):
        """ calculate p(r) function
        use the given i0 and rg value to fill in the low q part of the gap in data
        truncate the high q end at qmax
        """
        i_fs = get_font_size(fontsize)[0]
        if ax is None:
            ax = plt.gca()
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        if self.qgrid[-1] < qmax: qmax = self.qgrid[-1]
        tqgrid = np.arange(0, qmax, qmax / len(self.qgrid))
        tint = np.interp(tqgrid, self.qgrid, self.data)

        tint[tqgrid * rg < 1.] = i0 * np.exp(-(tqgrid[tqgrid * rg < 1.] * rg) ** 2 / 3.)
        # tint -= tint[-10:].sum()/10
        # Hanning window for reducing fringes in p(r)
        tw = np.hanning(2 * len(tqgrid) + 1)[len(tqgrid):-1]
        tint *= tw

        trgrid = np.arange(0, dmax, 1.)
        kern = np.asmatrix([[rj ** 2 * np.sinc(qi * rj / np.pi) for rj in trgrid] for qi in tqgrid])
        tt = np.asmatrix(tint * tqgrid ** 2).T
        tpr = np.reshape(np.array((kern.T * tt).T), len(trgrid))
        tpr /= tpr.sum()

        # plt.plot(tqgrid,tint,"g-")
        # tpr = np.fft.rfft(tint)
        # tx = range(len(tpr))
        ax.plot(trgrid, tpr, "g-")
        ax.set_xlabel("$r (\AA)$", fontsize=get_font_size(i_fs)[1])
        ax.set_ylabel("$P(r)$", fontsize=get_font_size(i_fs)[1])
        # plt.subplots_adjust(bottom=0.15)

    def save(self, fn, nz=True, scale_wabs=-1,
             save_comments=False, debug=False, footer=None):
        """
        should save all the relevant information, such as scaling, merging, averaging
        save data points with non-zero intensity only if nz==1
        """
        if scale_wabs>0:
            assert(self.trans_w>0)
            scale_wabs /= self.trans_w
            qidi = np.vstack((self.qgrid, self.data*scale_wabs, self.err*scale_wabs))
        else:
            qidi = np.vstack((self.qgrid, self.data, self.err))
        if nz:
            qidi = qidi[:, self.data != 0]
        if debug==True:
            print("saving file: %s, nz=%d" % (fn, nz))
        np.savetxt(fn, qidi.T, "%8.4f   %8.3e   %8.3e")
        if save_comments or footer is not None:
            ff = open(fn, "a")
            if save_comments:
                ff.write(self.comments)
                if scale_wabs>0:
                    ff.write("# converted to abs scale by applying a scaling factor of {scale_wabs:.2e}\n")
            elif footer is not None:
                ff.write(footer)
            ff.close()

    def plot(self, ax=None, scale=1., fontsize='large', scale_wabs=-1):
        i_fs = get_font_size(fontsize)[0]
        if ax is None:
            plt.figure()
            plt.subplots_adjust(bottom=0.15)
            ax = plt.gca()
        if scale_wabs>0:
            assert(self.trans_w>0)
            scale *= scale_wabs/self.trans_w
        ax.set_xlabel("$q (\AA^{-1})$", fontsize=get_font_size(i_fs)[1])
        ax.set_ylabel("$I$", fontsize=get_font_size(i_fs)[1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.errorbar(self.qgrid, self.data*scale, self.err*scale, label=self.label)
        for ov in self.overlaps:
            ax.plot(ov['q_overlap'], ov['raw_data1']*scale, "v")
            ax.plot(ov['q_overlap'], ov['raw_data2']*scale, "^")
        leg = ax.legend(loc='upper right', frameon=False)

        for t in leg.get_texts():
            t.set_fontsize(get_font_size(i_fs-2)[1])

def normalize(ds):
    return np.divide(ds.data, np.max(ds.data))

def calculate(ds0, ds1):
    diff_coef = np.sum(np.abs(np.subtract(ds0, ds1)))  # How different the datasets are
    return diff_coef            
            
def filter_by_similarity(datasets, similarity_threshold=0.5, debug=False):
    
    if len(datasets)==1:
        return datasets,None
    
    number_of_cpus = os.cpu_count()
    number_of_datasets = len(datasets)
    combinations = list(it.combinations(range(number_of_datasets), 2))
    similarity_matrix = np.zeros((number_of_datasets, number_of_datasets), dtype=np.bool)
    np.fill_diagonal(similarity_matrix, 1)  # If we compare the dataset with itself the result will always be one.

    norm_data = collections.deque(map(normalize, datasets))
    with mp.Pool(number_of_cpus) as pool:
        differences = pool.starmap(calculate, [(norm_data[i], norm_data[j]) for i, j in combinations])

    # print("Differences: \n", differences)
    # print("Diff Norm: \n", np.divide(differences, np.max(differences)))
    # print("Combinations: \n", combinations)
    similarities = np.divide(differences, np.max(differences)) <= similarity_threshold

    idx = 0
    for c in combinations:
        similarity_matrix[c[0]][c[1]] = similarities[idx]
        similarity_matrix[c[1]][c[0]] = similarities[idx]
        idx += 1

    number_of_simil_per_column = np.sum(similarity_matrix, axis=0)

    # No valid candidate, return all the data
    if np.array_equal(number_of_simil_per_column, np.ones(number_of_datasets)):
        if debug is True:
            print("No dataset with similarity level below threshold. Returning everything.")
        return datasets, []

    best_datasets_column = np.argmax(number_of_simil_per_column)
    best_column = similarity_matrix[:, best_datasets_column]
    valid_entries = list(it.compress(datasets, best_column))
    invalid_entries = set(datasets) - set(valid_entries)
    # print("Similarity Matrix: \n", similarity_matrix)
    # print("Best Column: \n", best_column)
    return valid_entries, invalid_entries


def merge_detectors(fns, detectors, qgrid, reft=-1, plot_data=False, save_ave=False, save_merged=False, ax=None, qmax=-1, qmin=-1, fix_scale=1, debug=False, transMode=trans_mode.from_waxs, trans=-1):
    """
    fns: filename, without the _SAXS/_WAXS suffix
    fix_scale is now default to 1
    implicitly assume that all detectors have the same qgrid
    """
    ss = []
    t0 = time.time()
    for fn in fns:
        s0 = Data1d()
        d_tot = np.zeros(qgrid.shape)
        d_max = np.zeros(qgrid.shape)
        d_min = np.zeros(qgrid.shape)+1.e32
        e_tot = np.zeros(qgrid.shape)
        c_tot = np.zeros(qgrid.shape)
        label = None
        comments = ""
        #t1 = time.time()
        for det in detectors:
            #t2 = time.time()
            # revised 2017mar10
            # old conversion: fn+det.extension gives the complete filename, the extension looks like this: "_SAXS.cbf"
            # this is a problem when the detector collect multiple images per trigger
            # new comvention: fn is a template, e.g. '/GPFS/xf16id/exp_path/301525/301016/temp1_000002%s_00001.cbf', 
            # and the extension looks like this: "_SAXS"  
            if "%s" in fn:
                fn1 = fn % det.extension
            else:
                fn1 = fn + det.extension
            if debug==True:
                print(fn, det.extension, fn1) 

            s0.load_from_2D(fn1, det.exp_para, qgrid, det.pre_process,
                            save_ave=save_ave, debug=debug)

            if save_ave:
                s0.save(fn1 + ".ave", debug=debug)

            if det.fix_scale is not None:
                fix_scale = det.fix_scale
                s0.scale(1./fix_scale)

            # empty part of the data is nan
            idx = ~np.isnan(s0.data)
            d_tot[idx] += s0.data[idx]
            e_tot[idx] += s0.err[idx]
            c_tot[idx] += 1

            idx1 = (np.ma.fix_invalid(s0.data, fill_value=-1)>d_max).data
            d_max[idx1] = s0.data[idx1]
            idx2 = (np.ma.fix_invalid(s0.data, fill_value=1e32)<d_min).data
            d_min[idx2] = s0.data[idx2]
            
            comments += s0.comments
            if label is None:
                label = s0.label
            else:
                label = common_name(label, s0.label)
        
        s0.data = d_tot
        s0.err = e_tot
        idx = (c_tot>1)
        s0.overlaps.append({'q_overlap': qgrid[idx],
                             'raw_data1': d_max[idx],
                             'raw_data2': d_min[idx]})
        s0.data[idx] /= c_tot[idx]
        s0.err[idx] /= c_tot[idx]
        s0.transMode = transMode
        s0.set_trans(transMode, trans=trans, ref_trans=reft, debug=debug)
        s0.label = label
        s0.comments = comments # .replace("# ", "## ")

        if save_merged:
            s0.save(s0.label + ".dd", debug=debug)
        ss.append(s0)

    return ss


def average(fns, detectors, qgrid, reft=-1, plot_data=False, save1d=0, ax=None, qmax=-1, qmin=-1, fix_scale=-1,
            filter_datasets=True, similarity_threshold=0.5, debug=False):
    """
    fns: filename, without the _SAXS/_WAXS suffix
    save1d:  0 = do not save 1d data; 1 = save only the averaged data *.ddd;
             2 = merged data before average as well, *.dd; 3 = also save the data before merge, *.ave 
    """

    save_dd = False
    save_ave = False
    save_ddd = False
    if save1d>0:
        save_ddd = True
    if save1d>1: 
        save_dd = True
    if save1d>2:
        save_ave = True

    t0 = time.time()
    ss = merge_detectors(fns, detectors, qgrid, reft, plot_data, save_ave, save_dd, 
                         ax, qmax, qmin, fix_scale, debug=debug)
    t1 = time.time()
    if filter_datasets:
        ss, invalids = filter_by_similarity(ss, similarity_threshold=similarity_threshold, debug=debug)
        # TODO: Insert warning/exception when the number of datasets discarded is high.
        # TODO: Define the % of discarded to result in a error.
        if debug!='quiet':
            print("Selected Datasets: ")
            for s in ss:
                print(s.label)

        if len(invalids) > 0 and debug!='quiet':
            print("The following datasets where discarded due to similarity level below the threshold: ",
                  similarity_threshold)
            for inv in invalids:
                print(inv.label)
    t2 = time.time()
    if len(ss) > 0:
        ss[0] = ss[0].avg(ss[1:], plot_data, ax=ax, debug=debug)
    t3 = time.time()
    if save_ddd:
        ss[0].save(ss[0].label + ".ddd", debug=debug)
    t4 = time.time()
    if debug==True:
        print('Time to Merge: ', t1-t0)
        print('Time to Filter: ', t2-t1)
        print('Time to Average: ', t3-t2)
        print('Time to Save Data: ', t4-t3)
    return ss[0]


def process(sfns, bfns, detectors, qgrid, qmax=-1, qmin=-1, 
            reft=-1, save1d=False, conc=0., plot_data=True, fix_scale=-1,
            filter_datasets=True, similarity_threshold=0.5, debug=False):
    vfrac = 0.001 * conc / PROTEIN_WATER_DENSITY_RATIO

    sample_axis = None
    buffer_axis = None
    bkg_cor_axis = None

    if plot_data:
        plt.figure()
        sample_axis = plt.subplot2grid((2, 2), (0, 0), title="Sample Data")
        buffer_axis = plt.subplot2grid((2, 2), (0, 1), title="Buffer Data")
        bkg_cor_axis = plt.subplot2grid((2, 2), (1, 0), colspan=2, title="Background Correction")

    # TODO: Run the next two lines in parallel
    args_sample = (sfns, detectors, qgrid, reft, plot_data, save1d, sample_axis, qmax,
                   qmin, fix_scale, filter_datasets, similarity_threshold)
    args_buffer = (bfns, detectors, qgrid, reft, plot_data, save1d, buffer_axis, qmax,
                   qmin, fix_scale, filter_datasets, similarity_threshold)

    # In order for the Pool to work we need to make the ExpPara something other than a SwigObject.
    # Maybe it is time to change for something else
    # with mp.Pool(1) as pool:
    #     result = pool.starmap(average, (args_sample, args_buffer))
    #
    # ds = result[0]
    # db = result[1]

    ds = average(*args_sample, debug=debug)
    db = average(*args_buffer, debug=debug)

    ds.bkg_cor(db, 1.0 - vfrac, plot_data=plot_data, ax=bkg_cor_axis, inplace=True, debug=debug)

    return ds


def analyze(d1, qstart, qend, fix_qe, qcutoff, dmax):
    plt.figure(figsize=(14, 5.5))
    plt.subplot(121)
    I0,Rg,_ = d1.plot_Guinier(qs=qstart, qe=qend, fix_qe=fix_qe)

    print("I0=%f, Rg=%f" % (I0, Rg))

    plt.subplot(122)
    d1.plot_pr(I0, Rg, qmax=1.2, dmax=dmax)
    plt.subplots_adjust(bottom=0.15, wspace=0.25)

    
def estimate_scaling_factor(d1s, d1b, 
                            q_min=0.5, q_max=3.5, smoothing_width=5, prec=4, s_thresh=1,
                            plot_data=False, ax=None, debug=False):
    """ Estimate the scaling factor needed to subtract buffer scattering d1b
        from sample scattering d1s, d1s/d1b should be instances of Data1d
        
        This function iteratively vary the scaling factor, up to the specified 
        precision (number of digits after the decimal point), and optinally smooth 
        the data before the calcualtion. Several critiera for over sub-traction are 
        used to stop the interation: 
        (1) non-zero value in the subtracted result
        (2) minimum of the subtracted result fall below 1/q
        (3) dynamic range of the value of the result, as measured by the span or std
            deviation of the log, becomes too high
        
        only test the data above q_min, which should not exceed 1.0
    """
    idx = (d1s.qgrid>q_min) & (d1s.qgrid<q_max) & (d1s.data>0) & (d1b.data>0)
    md1s = d1s.data[idx]
    md1b = d1b.data[idx]
    try:
        md1sm = smooth(md1s, half_window_len=smoothing_width)
        md1bm = smooth(md1b, half_window_len=smoothing_width)
    except:
        return 1.0      # problems like incomplete data can cause smooth() to fail, no point to continue
    mq = d1s.qgrid[idx]
    sc = 0.9
    prec0 = 2
    std0 = np.log(md1s-md1b*sc).std()
    n,bins = np.histogram((md1s-md1b*sc)*mq)
    sp0 = bins[-1]-bins[0]
    
    if debug:
        print("# sc,   mean,   std,   span,   qmin")
    while prec0<=prec:
        sc1 = sc
        while np.all(md1s-sc1*md1b>0):
            # data should have finite dynamic range
            td = np.log(md1s-md1b*sc1)
            n,bins = np.histogram(td)
            sp1 = bins[-1]-bins[0]
            std1 = td.std()
            #if std1>s_thresh*std0:
            #    break
            #elif std1<std0:
            #    std0 = std1
            if sp1>sp0+s_thresh:
                if debug:
                    print(f"span exceeded threshold: {sp1}, {sp0}+{s_thresh}")
                break
            # assume that data*q should have a lower bound
            # this should work better on the smoothed data
            td1 = (md1sm-md1bm*sc1)*mq
            q_Imin = mq[td1.argmin()]
            if q_Imin>1.7:  # under the water peak, indication of over-subtraction
                if debug:
                    print(f"qxI min under water peak: q={q_Imin:.3f}")
                break
            sc = sc1
            if debug:
                print(f"{sc1:.5f}, {td.mean():.3f}, {std1:.3f}, {sp1:3f}, {q_Imin:.3f}")
            sc1 = sc+np.power(10., -prec0)
        prec0 += 1
    
    if plot_data:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        plt.semilogy(d1s.qgrid, d1s.data)
        plt.semilogy(d1b.qgrid, d1b.data)
        plt.errorbar(d1s.qgrid, (d1s.data-sc*d1b.data), d1s.err*1.414)
    
    return sc
    
