import ipywidgets
from IPython.display import display,clear_output
import numpy as np
from py4xs.hdf import h5sol_HT,lsh5
from py4xs.slnxs import get_font_size
import pylab as plt
from io import StringIO
import copy,subprocess

def display_solHT_data(fn, atsas_path=""):
    """ atsas_path for windows might be c:\atsas\bin
    """
    dt = h5sol_HT(fn)
    dt.load_d1s()
    dt.subtract_buffer(sc_factor=-1, debug='quiet')
    
    # widgets
    ddSample = ipywidgets.Dropdown(options=dt.samples, 
                                   value=dt.samples[0], description='Sample:')
    sampleLabels = [f"frame #{i}" for i in range(len(dt.attrs[dt.samples[0]]['selected']))]
    smAverage = ipywidgets.SelectMultiple(options=sampleLabels, 
                                          layout=ipywidgets.Layout(width='10%'),
                                          descripetion="selection for averaging")
    
    btnExport = ipywidgets.Button(description='Export', 
                                  layout=ipywidgets.Layout(width='20%'), 
                                  style = {'description_width': 'initial'})
    exportSubtractedCB = ipywidgets.Checkbox(value=True, description='export subtracted',
                                             #layout=ipywidgets.Layout(width='30%'),
                                             style = {'description_width': 'initial'})
    btnUpdate = ipywidgets.Button(description='Update plot', layout=ipywidgets.Layout(width='35%'))

    subtractCB = ipywidgets.Checkbox(value=False, 
                                     style = {'description_width': 'initial'},
                                     description='show subtracted')

    slideScFactor = ipywidgets.FloatSlider(value=1.0, min=0.8, max=1.2, step=0.001,
                                           style = {'description_width': 'initial'},
                                           description='Scaling factor:', readout_format='.3f')
    guinierQsTx = ipywidgets.Text(value='0.01', 
                                  layout=ipywidgets.Layout(width='40%'),
                                  description='Guinier fit qs:')
    guinierRgTx = ipywidgets.Text(value='', 
                                  layout=ipywidgets.Layout(width='35%'), 
                                  description='Rg:')

    vbox1 = ipywidgets.VBox([ddSample, slideScFactor, 
                             ipywidgets.HBox([guinierQsTx, guinierRgTx])],
                            layout=ipywidgets.Layout(width='40%'))
    vbox2 = ipywidgets.VBox([ipywidgets.HBox([btnExport, exportSubtractedCB]), 
                             btnUpdate, subtractCB], 
                            layout=ipywidgets.Layout(width='45%'))
    hbox1 = ipywidgets.HBox([vbox1, smAverage, vbox2])                

    btnReport = ipywidgets.Button(description='ATSAS report') #, layout=ipywidgets.Layout(width='20%'))
    qSkipTx = ipywidgets.Text(value='0', description='skip:', 
                                layout=ipywidgets.Layout(width='20%'),
                                style = {'description_width': 'initial'})    
    qCutoffTx = ipywidgets.Text(value='0.3', description='q cutoff:', 
                                layout=ipywidgets.Layout(width='25%'),
                                style = {'description_width': 'initial'})    
    outTxt = ipywidgets.Textarea(layout=ipywidgets.Layout(width='55%', height='100%'))
    hbox5 = ipywidgets.HBox([outTxt, 
                             ipywidgets.VBox([btnReport, 
                                              ipywidgets.HBox([qSkipTx, qCutoffTx])]) ])    
    
    box = ipywidgets.VBox([hbox1, hbox5])
    display(box)
    fig1 = plt.figure(figsize=(7, 4))
    # rect = l, b, w, h
    ax1 = fig1.add_axes([0.1, 0.15, 0.5, 0.78])
    ax2 = fig1.add_axes([0.72, 0.61, 0.26, 0.32])
    ax3 = fig1.add_axes([0.72, 0.15, 0.26, 0.32])

    axr = []
    fig2 = plt.figure(figsize=(7,2.5))
    axr.append(fig2.add_axes([0.09, 0.25, 0.25, 0.6])) 
    axr.append(fig2.add_axes([0.41, 0.25, 0.25, 0.6])) 
    axr.append(fig2.add_axes([0.73, 0.25, 0.25, 0.6])) 
                                 
    def onChangeSample(w):
        sn = ddSample.value
        sel = [sampleLabels[i] for i in range(len(sampleLabels)) 
               if dt.attrs[sn]['selected'][i]]
        smAverage.value = sel    
        isSample = ('sc_factor' in dt.attrs[sn].keys())
        for a in axr:
            a.clear()
        outTxt.value = ""

        if isSample:
            subtractCB.disabled = False
            slideScFactor.value = dt.attrs[sn]['sc_factor']
            exportSubtractedCB.disabled = False
            if subtractCB.value:
                btnReport.disabled = False
        else:
            subtractCB.value = False
            subtractCB.disabled = True
            slideScFactor.disabled = True
            exportSubtractedCB.value = False
            exportSubtractedCB.disabled = True
            btnReport.disabled = True
        onUpdatePlot(None)
    
    def onReport(w):
        #try:
        txt = gen_report_d1s(dt.d1s[ddSample.value]["subtracted"], ax=axr, 
                             skip=int(qSkipTx.value), q_cutoff=float(qCutoffTx.value), 
                             print_results=False, path=atsas_path)
        outTxt.value = txt
        #except:
        #    outTxt.value = "unable to run ATSAS ..."
    
    def onUpdatePlot(w):
        sn = ddSample.value
        re_calc = False
        show_sub = subtractCB.value
        sc_factor = slideScFactor.value
        sel = [(sampleLabels[i] in smAverage.value) for i in range(len(sampleLabels))]
        isSample = ('sc_factor' in dt.attrs[sn].keys())
        if w is not None:
            if np.any(sel != dt.attrs[sn]['selected']):
                dt.average_d1s(sn, selection=sel, debug=False)
                if isSample:
                    re_calc = True
            if isSample:
                if sc_factor!=dt.attrs[sn]['sc_factor']:
                    re_calc = True
            if re_calc:
                dt.subtract_buffer(sn, sc_factor=sc_factor, debug='quiet')
                re_calc = False
        ax1.clear()
        dt.plot_sample(sn, ax=ax1, show_subtracted=show_sub)
        ax2.clear()
        ax3.clear()
        if isSample and show_sub:
            d1 = dt.d1s[sn]['subtracted']
            ym = np.max(d1.data[d1.qgrid>0.5])
            qm = d1.qgrid[d1.data>0][-1]
            ax2.semilogy(d1.qgrid, d1.data)
            ax2.errorbar(d1.qgrid, d1.data, d1.err)
            ax2.set_xlim(left=0.5, right=qm)
            ax2.set_ylim(top=ym*1.1)
            ax2.yaxis.set_major_formatter(plt.NullFormatter())
            qs = np.float(guinierQsTx.value)
            i0,rg = dt.d1s[sn]['subtracted'].plot_Guinier(ax=ax3, qs=qs, fontsize=0)
            ax3.yaxis.set_major_formatter(plt.NullFormatter())
            guinierRgTx.value = ("%.2f" % rg)
            #print(f"I0={i0}, Rg={.2f:rg}")
            #plt.tight_layout()
            ax2.set_title("buf subtraction")
            ax3.set_title("Guinier")
    
    def onShowSubChanged(w):
        show_sub = subtractCB.value
        if show_sub:
            slideScFactor.disabled = False
            smAverage.disabled = True
            btnReport.disabled = False
        else:
            slideScFactor.disabled = True
            smAverage.disabled = False
            btnReport.disabled = True
        onUpdatePlot(None)
    
    def onExport(w):
        sn = ddSample.value
        dt.export_d1s(sn, save_subtracted=exportSubtractedCB.value)
        dt.update_h5()
        
    onChangeSample(None)
    btnUpdate.on_click(onUpdatePlot)
    subtractCB.observe(onShowSubChanged)
    slideScFactor.observe(onUpdatePlot)
    ddSample.observe(onChangeSample)
    btnExport.on_click(onExport)
    btnReport.on_click(onReport)
    
    return dt

def run(cmd, path=""):
    cmd = path+cmd
    p = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if len(err)>0:
        print(err.decode())
        raise Exception(err.decode())
    return out.decode()

def extract_vals(txt, dtype=float, strip=None, debug=False):
    if strip is not None:
        txt = txt.replace(strip, " ")
    sl = txt.split(" ")
    ret = []
    for ss in sl:
        try:
            val = dtype(ss)
        except:
            pass
        else:
            ret.append(val)
    return ret
    
def atsas_create_temp_file(fn, d1s, skip=0, q_cutoff=0.6):
    idx = (d1s.qgrid<=q_cutoff)
    np.savetxt(fn, np.vstack([d1s.qgrid[idx][skip:], d1s.data[idx][skip:], d1s.err[idx][skip:]]).T)
    
def atsas_autorg(fn, debug=False, path=""):
    ret = run(f"autorg {fn}", path).split('\n')
    #rg,drg = extract_vals(ret[0], "+/-", debug=debug)
    #i0,di0 = extract_vals(ret[1], "+/-", debug=debug)
    #n1,n2 = extract_vals(ret[2], " to ", debug=debug, dtype=int)
    #qual = extract_vals(ret[3], "%", debug=debug)
    rg,drg = extract_vals(ret[0])
    i0,di0 = extract_vals(ret[1])
    n1,n2 = extract_vals(ret[2], dtype=int)
    qual = extract_vals(ret[3], strip="%")[0]
    
    return {"Rg": rg, "Rg err": drg, 
            "I0": i0, "I0 err": di0,
            "fit range": [n1,n2],
            "quality": qual}

def atsas_datgnom(fn, rg, first, last, fn_out=None, path=""):
    """ 
    """
    if fn_out is None:
        fn_out = fn.split('.')[0]+'.out'
    
    options = f"-r {rg} --first {first} --last {last} -o {fn_out}"
    # datgnom vs datgnom4, slightly different input parameters
    ret = run(f"datgnom {options} {fn}", path).split("\n")
    dmax,qual = extract_vals(ret[0])
    rgg,rgp = extract_vals(ret[1])
    
    return {"Dmax": dmax, "quality": qual, 
            "Rg (q)": rgg, "Rg (r)": rgp}

def read_arr_from_strings(lines, cols=[0,1,2]):
    """ assuming that any none numerical values will be ignored
        data are in multiple columns
        some columns may be missing values at the top
        for P(r), cols=[0,1,2]
        for I_fit(q), cols=[0,-1]
    """
    ret = []
    for buf in lines:
        if len(buf)<len(cols):  # empty line
            continue        
        tb = np.genfromtxt(StringIO(buf))
        if np.isnan(tb).any():   # mixed text and numbersS          J EXP       ERROR       J REG       I REG
            continue
        ret.append([tb[i] for i in cols])
    return np.asarray(ret).T

def read_gnom_out_file(fn, plot_pr=False, ax=None):
    ff = open(fn, "r")
    tt = ff.read()
    ff.close()
    
    hdr,t1 = tt.split("####      Experimental Data and Fit                     ####")
    #hdr,t1 = tt.split("S          J EXP       ERROR       J REG       I REG")
    iq, pr = t1.split("####      Real Space Data                               ####")
    #iq, pr = t1.split("Distance distribution  function of particle")
    di, dq = read_arr_from_strings(iq.rstrip().split("\n"), cols=[0,-1])
    dr, dpr, dpre = read_arr_from_strings(pr.rstrip().split("\n"), cols=[0,1,2])
    
    if plot_pr:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.errorbar(dr, dpr, dpre)
    
    return hdr.rstrip(),di,dq,dr,dpr,dpre

# ALMERGE   Automatically merges data collected from two different concentrations or 
#           extrapolates it to infinite dilution assuming moderate particle interactions.
#

def atsas_dat_tools(fn_out, path=""):
    # datporod: the used Rg, I0, the computed volume estimate and the input file name
    #
    # datvc: the first three numbers are the integrated intensities up to 0.2, 0.25 and 0.3, respectively. 
    #        the second three numbers the corresponding MW estimates
    #
    # datmow: Output: Q', V' (apparent Volume), V (Volume, A^3), MW (Da), file name
    ret = run(f"datporod {fn_out}", path).split('\n')
    t,t,Vv = extract_vals(ret[0])
    r_porod = {"vol": Vv}
    
    #ret = run(f"datvc {fn_out}").split('\n')
    #ii1,ii2,ii3,mw1,mw2,mw3 = extract_vals(ret[0])
    #r_vc = {"MW": [mw1, mw2, mw3]}
    
    ret = run(f"datmow {fn_out}", path).split('\n')
    Qp,Vp,Vv,mw = extract_vals(ret[0])
    r_mow = {"Q'": Qp, "app vol": Vp, "vol": Vv, "MW": mw}

    return {"datporod": r_porod, 
            #"datvc": r_vc,  # this won't work if q_max is below 0.3 
            "datmow": r_mow}
    
def gen_report_d1s(d1s, ax=None, skip=0, q_cutoff=0.6, print_results=True, path=""):
    if ax is None:
        ax = []
        fig = plt.figure(figsize=(9,3))
        # rect = l, b, w, h
        ax.append(fig.add_axes([0.09, 0.25, 0.25, 0.6])) 
        ax.append(fig.add_axes([0.41, 0.25, 0.25, 0.6])) 
        ax.append(fig.add_axes([0.75, 0.25, 0.25, 0.6])) 
    else:
        for a in ax:
            a.clear()
    
    tfn = "t.dat"
    tfn_out = "t.out"
    atsas_create_temp_file(tfn, d1s, skip=skip, q_cutoff=q_cutoff)

    re_autorg = atsas_autorg(tfn, path=path)
    re_gnom = atsas_datgnom(tfn, re_autorg["Rg"], first=skip+1,
                            last=len(d1s.qgrid[d1s.qgrid<=q_cutoff]), fn_out=tfn_out, path=path)
    hdr,di,dq,dr,dpr,dpre = read_gnom_out_file(tfn_out)
    
    idx = (d1s.qgrid<q_cutoff)
    ax[0].semilogy(di, dq)
    ax[0].errorbar(d1s.qgrid[idx], d1s.data[idx], d1s.err[idx])
    #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
    ax[0].set_title("intensity")
    ax[0].set_xlabel("q")

    if re_autorg["Rg"]==0:
        kratky_qm=0.3
        idx = (d1s.qgrid<kratky_qm)
        ax[1].plot(d1s.qgrid[idx], d1s.data[idx]*np.power(d1s.qgrid[idx], 2))
        ax[1].set_xlabel("q")
    else:
        kratky_qm=10./re_autorg["Rg"]
        idx = (d1s.qgrid<kratky_qm)
        ax[1].plot(d1s.qgrid[idx]*re_autorg["Rg"], d1s.data[idx]*np.power(d1s.qgrid[idx], 2))
        ax[1].set_xlabel("q x Rg")    
    ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_title("kratky plot")

    ax[2].errorbar(dr, dpr, dpre)
    ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    ax[2].set_title("P(r)")
    ax[2].set_xlabel("r")

    ret = atsas_dat_tools(tfn_out, path=path)
    if print_results:
        print(f"Gunier fit: quality = {re_autorg['quality']} %,", end=" ")
        print(f"I0 = {re_autorg['I0']:.2f} +/- {re_autorg['I0 err']:.2f} , ", end="")
        print(f"Rg = {re_autorg['Rg']:.2f} +/- {re_autorg['Rg err']:.2f}")
        print(f"GNOM fit: quality = {re_gnom['quality']:.2f}, Dmax = {re_gnom['Dmax']:.2f}, Rg = {re_gnom['Rg (r)']:.2f}")
        print(f"Volume estimate: {ret['datporod']['vol']:.1f} (datporod), {ret['datmow']['vol']:.1f} (MoW)")
        print(f"MW estimate: {ret['datmow']['MW']/1000:.1f} kDa (MoW)")          
    else:
        txt = f"Gunier fit: quality = {re_autorg['quality']} %, "
        txt += f"I0 = {re_autorg['I0']:.2f} +/- {re_autorg['I0 err']:.2f} , "
        txt += f"Rg = {re_autorg['Rg']:.2f} +/- {re_autorg['Rg err']:.2f}\n"
        txt += f"GNOM fit: quality = {re_gnom['quality']:.2f}, Dmax = {re_gnom['Dmax']:.2f}, Rg = {re_gnom['Rg (r)']:.2f}\n"
        txt += f"Volume estimate: {ret['datporod']['vol']:.1f} (datporod), {ret['datmow']['vol']:.1f} (MoW)\n"
        txt += f"MW estimate: {ret['datmow']['MW']/1000:.1f} kDa (MoW)"          
        return txt