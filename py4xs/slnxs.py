import numpy as np
import matplotlib.pyplot as plt
import collections

from py4xs.data2d import Data2d
from py4xs.mask import Mask
from py4xs.utils import common_name

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

# trans_mode=TRANS_FROM_BEAM_CENTER
BEAM_SIZE_hW = 5
BEAM_SIZE_hH = 4

trans_mode = TRANS_FROM_WAXS
# this is the minimum intensity to be used for trans calculations
WAXS_THRESH = 100

# this is the scaling factor for indivudual curves that belong to the same sample
# they are offset for clarity in the plots
VOFFSET = 1.5


class Data1d:
    def __init__(self):
        self.comments = ""
        self.label = "data"
        self.overlaps = []
        self.raw_data = {}
        self.timestamp = None
        
    def load_from_2D(self, image, exp_para, qgrid, pre_process, 
                     mask=None, save_ave=False, debug=False, label=None):
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
        if isinstance(image, str):
            d2 = Data2d(image, exp=exp_para)
            self.comments += "# loaded from %s\n" % image
            self.label = image.split("/")[-1]
            self.timestamp = d2.timestamp
        elif isinstance(image, np.ndarray):
            d2 = Data2d(None, im=image, exp=exp_para)
            if label is not None:
                self.label = label
        elif isinstance(image, Data2d):
            d2 = image
            if label is not None:
                self.label = label
        else:
            raise Exception("unable to create load data from image:", image)
            
        # deal with things like dark current, flat field, and dezinger corrections on the 2D data
        pre_process(d2.data)
        
        if trans_mode == TRANS_FROM_BEAM_CENTER:
            # get trans from beam center
            self.roi = d2.data.val(exp_para.bm_ctr_x-BEAM_SIZE_hW, 
                                   exp_para.bm_ctr_x+BEAM_SIZE_hW,
                                   exp_para.bm_ctr_y-BEAM_SIZE_hH,
                                   exp_para.bm_ctr_y+BEAM_SIZE_hH, mask)

        self.data,self.err = d2.conv_Iq(qgrid, mask,
                                        cor_factor = exp_para.FPol)  # exp_para.FSA*exp_para.FPol)
        if isinstance(image, np.ndarray):
            del d2      # d2 is only used temporarily
        
        if save_ave and isinstance(image, str):
            self.save(image + ".ave", debug=debug)     
        

    def set_trans(self, trans=-1, ref_trans=-1, debug=False):
        """
        normalize intensity, from trans to ref_trans
        trans can be either from the beam center or water scattering
        this operation should be performed after SAXS/WAXS merge, because
        1. SAXS and WAXS should have the same trans
        2. if trans_mode is TRNAS_FROM_WAXS, the trans value needs to be calculated from WAXS data
        """
        if trans_mode == TRANS_FROM_BEAM_CENTER:
            # get trans from beam center
            self.trans = self.roi
            self.comments += "# transmitted beam intensity from beam center, "
        elif trans_mode == TRANS_FROM_WAXS:
            # get trans for the near the maximum in the WAXS data
            # for solution scattering, hopefully this reflect the intensity of water scattering
            idx = (self.qgrid > 1.85) & (self.qgrid < 2.15)  # & (self.data>0.5*np.max(self.data))
            if len(self.qgrid[idx]) < 5:
                print("not enough data points under the water peak, consider using a different trans_mode.")
                #raise Exception()
            idx1 = idx & (self.data > 0.95*np.max(self.data[idx]))
            if (self.data[idx1]<WAXS_THRESH).all() and debug!='quiet':
                print("the data points for trans calculation are below WAXS_THRESH: ", 
                      np.max(self.data[idx1]), WAXS_THRESH)                
            self.trans = np.sum(self.data[idx1])
            qavg = np.average(self.qgrid[idx1])
            if debug==True:
                print("using data near the high q end (q~%f)" % qavg, end=' ')
            self.comments += "# transmitted beam intensity from WAXS (q~%.2f)" % qavg
        elif trans_mode == TRANS_EXTERNAL:
            if trans <= 0:
                print("trans_mode is TRANS_EXTERNAL but trans value is not provided")
                exit()
            self.comments += "# transmitted beam intensity is defined externally"
            self.trans = trans
        else:
            print("invalid transmode: ", trans_mode)

        self.comments += ": %f \n" % self.trans
        if debug==True:
            print("trans for %s set to %f" % (self.label, self.trans))

        if ref_trans > 0:
            self.comments += "# scattering intensity normalized to ref_trans = %f \n" % ref_trans
            self.data *= ref_trans / self.trans
            self.err *= ref_trans / self.trans
            for ov in self.overlaps:
                ov['raw_data1'] *= ref_trans / self.trans
                ov['raw_data2'] *= ref_trans / self.trans
            self.trans = ref_trans
            if debug==True:
                print("normalized to %f" % ref_trans)


    def avg(self, dsets, plot_data=False, ax=None, debug=False):
        """
        dset is a collection of Data1d
        ax is the Axes to plot the data in
        TODO: should calculate something like the cross-correlation between sets
        to evaluate the consistency between them
        """
        if debug!='quiet':
            print("averaging data with %s: \n" % self.label, end=' ')
 
        n = 1
        if plot_data:
            if ax is None:
                plt.figure()
                plt.subplots_adjust(bottom=0.15)
                ax = plt.gca()
            ax.set_xlabel("$q (\AA^{-1})$", fontsize='x-large')
            ax.set_ylabel("$I$", fontsize='x-large')
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
        for d1 in dsets:
            if debug==True:
                print("%s " % d1.label, end=' ')
            if not (d0.qgrid == d1.qgrid).all():
                raise Exception("\n1D sets cannot be averaged: qgrid mismatch")
            d0.trans += d1.trans
            d0.data += d1.data
            d0.err += d1.err
            if trans_mode == TRANS_FROM_BEAM_CENTER:
                d0.roi += d1.roi
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
        d0.data /= n
        d0.err /= np.sqrt(n)
        if trans_mode == TRANS_FROM_BEAM_CENTER:
            d0.roi /= n
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
                t.set_fontsize('small')

        return d0


    def bkg_cor(self, dbak, sc_factor=1., plot_data=False, ax=None, inplace=False, debug=False):
        """
        background subtraction
        """
        dset = None
        if inplace:
            dset = self
        else:
            dset = copy.deepcopy(self)

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
            ax.set_xlabel("$q (\AA^{-1})$", fontsize='x-large')
            ax.set_ylabel("$I$", fontsize='x-large')
            ax.set_xscale('log')
            ax.set_yscale('log')
            idx = (dset.data > 0) & (dbak.data > 0)
            ax.plot(dset.qgrid[idx], dset.data[idx], label=self.label)
            ax.plot(dbak.qgrid[idx], dbak.data[idx], label=dbak.label)
            ax.plot(dbak.qgrid[idx], dbak.data[idx] * sc * sc_factor, label=dbak.label + ", scaled")

        if len(dset.overlaps) != len(dbak.overlaps):
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
                t.set_fontsize('small')

        if debug==True:
            print("using scaling factor of %f" % (sc * sc_factor))
        dset.data -= dbak.data * sc * sc_factor
        dset.err += dbak.err * sc * sc_factor
        if plot_data:
            ax.errorbar(dset.qgrid, dset.data, dset.err)

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
        self.comments += "# data is scaled by %f.\n" % sc
        if len(self.overlaps) != 0:
            for ov in self.overlaps:
                ov['raw_data1'] *= sc
                ov['raw_data2'] *= sc

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
            self.data[idx] = (self.data[idx] + d1.data[idx]) / 2
            # this won't work well if the merging data are mis-matched before bkg subtraction
            # but match well after bkg subtraction
            self.err[idx] = (self.err[idx] + d1.err[idx]) / 2
        self.data[self.qgrid >= qmax] = d1.data[self.qgrid >= qmax]
        self.err[self.qgrid >= qmax] = d1.err[self.qgrid >= qmax]

        self.comments += "# merged with the following set by matching intensity within (%.4f, %.4f)," % (qmin, qmax)
        self.comments += " scaled by %f\n" % sc
        self.comments += d1.comments.replace("# ", "## ")

    def plot_Guinier(self, qs=0, qe=10, rg=15, fix_qe=False, ax=None, no_plot=False):
        """ do Gunier plot, estimate Rg automatically
        qs specify the lower end of the q-range to perform the fit in
        rg is the optinal initial estimate
        if fix_qe==1, qe defined the end of the region to perform the fit
        """
        idx = (self.data > 0)
        # print self.data

        if no_plot==False:
            if ax is None:
                ax = plt.gca()
            ax.set_xscale('linear')
            ax.set_yscale('log')
            ax.errorbar(self.qgrid[idx] ** 2, self.data[idx], self.err[idx])

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
            rg = np.sqrt(-rg * 3.)
            cnt += 1
            # print i0, rg
        td[1, :] = i0 * np.exp(-td[0, :]*rg*rg/3.)

        if no_plot==False:
            ax.plot([td[0, 0], td[0, -1]], [td[1, 0], td[1, -1]], "ro")
            ax.plot(self.qgrid ** 2, i0 * np.exp(-self.qgrid ** 2 * rg * rg / 3.))
            ax.set_ylabel("$I$", fontsize='x-large')
            ax.set_xlabel("$q^2 (\AA^{-2})$", fontsize='x-large')
            # plt.subplots_adjust(bottom=0.15)
            ax.set_xlim(0, qe * qe * 2.)
            ax.autoscale_view(tight=True, scalex=False, scaley=True)
            ax.set_ylim(ymin=i0 * np.exp(-qe * qe * 2. * rg * rg / 3.))

        # print "I0=%f, Rg=%f" % (i0,rg)
        return (i0, rg)

    def plot_pr(self, i0, rg, qmax=5., dmax=200., ax=None):
        """ calculate p(r) function
        use the given i0 and rg value to fill in the low q part of the gap in data
        truncate the high q end at qmax
        """
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
        ax.set_xlabel("$r (\AA)$", fontsize='x-large')
        ax.set_ylabel("$P(r)$", fontsize='x-large')
        # plt.subplots_adjust(bottom=0.15)

    def save(self, fn, nz=True, save_comments=False, debug=False):
        """
        should save all the relevant information, such as scaling, merging, averaging
        save data points with non-zero intensity only if nz==1
        """
        qidi = np.vstack((self.qgrid, self.data, self.err))
        if nz:
            qidi = qidi[:, self.data != 0]
        if debug==True:
            print("saving file: %s, nz=%d" % (fn, nz))
        np.savetxt(fn, qidi.T, "%12.5f")
        if save_comments:
            ff = open(fn, "a")
            ff.write(self.comments)
            ff.close()

    def plot(self, ax=None, scale=1.):
        if ax is None:
            plt.figure()
            plt.subplots_adjust(bottom=0.15)
            ax = plt.gca()
        ax.set_xlabel("$q (\AA^{-1})$", fontsize='x-large')
        ax.set_ylabel("$I$", fontsize='x-large')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.errorbar(self.qgrid, self.data*scale, self.err*scale, label=self.label)
        for ov in self.overlaps:
            ax.plot(ov['q_overlap'], ov['raw_data1']*scale, "v")
            ax.plot(ov['q_overlap'], ov['raw_data2']*scale, "^")
        leg = ax.legend(loc='upper right', frameon=False)

        for t in leg.get_texts():
            t.set_fontsize('small')


def normalize(ds):
    return np.divide(ds.data, np.max(ds.data))

def calculate(ds0, ds1):
    diff_coef = np.sum(np.abs(np.subtract(ds0, ds1)))  # How different the datasets are
    return diff_coef            
            
def filter_by_similarity(datasets, similarity_threshold=0.5):
    
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
        print("No dataset with similarity level below threshold. Returning everything.")
        return datasets, []

    best_datasets_column = np.argmax(number_of_simil_per_column)
    best_column = similarity_matrix[:, best_datasets_column]
    valid_entries = list(it.compress(datasets, best_column))
    invalid_entries = set(datasets) - set(valid_entries)
    # print("Similarity Matrix: \n", similarity_matrix)
    # print("Best Column: \n", best_column)
    return valid_entries, invalid_entries


def merge_detectors(fns, detectors, qgrid, reft=-1, plot_data=False, save_ave=False, save_merged=False, ax=None, qmax=-1, qmin=-1, fix_scale=1, debug=False):
    """
    fns: filename, without the _SAXS/_WAXS suffix
    fix_scale is now default to 1
    implicitly assume that all detectors have the same qgrid
    """
    ss = []
    t0 = time.time()
    for fn in fns:
        s0 = Data1d()
        d_tot = np.zeros(detectors[0].qgrid.shape)
        d_max = np.zeros(detectors[0].qgrid.shape)
        d_min = np.zeros(detectors[0].qgrid.shape)+1.e32
        e_tot = np.zeros(detectors[0].qgrid.shape)
        c_tot = np.zeros(detectors[0].qgrid.shape)
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
        s0.overlaps.append({'q_overlap': detectors[0].qgrid[idx],
                             'raw_data1': d_max[idx],
                             'raw_data2': d_min[idx]})
        s0.data[idx] /= c_tot[idx]
        s0.err[idx] /= c_tot[idx]
        s0.set_trans(ref_trans=reft, debug=debug)
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
        ss, invalids = filter_by_similarity(ss, similarity_threshold=similarity_threshold)
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
    (I0, Rg) = d1.plot_Guinier(qs=qstart, qe=qend, fix_qe=fix_qe)

    print("I0=%f, Rg=%f" % (I0, Rg))

    plt.subplot(122)
    d1.plot_pr(I0, Rg, qmax=1.2, dmax=dmax)
    plt.subplots_adjust(bottom=0.15, wspace=0.25)


