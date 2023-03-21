#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint
from distributions import distributions, binnings

pjoin = os.path.join

legend_labels = {
    'DY.*' : "QCD Z$\\rightarrow\\ell\\ell$",
    'EWKZ.*ZToLL.*' : "EWK Z$\\rightarrow\\ell\\ell$",
    'WN*J.*LNu.*' : "QCD W$\\rightarrow\\ell\\nu$",
    'EWKW.*LNu.*' : "EWK W$\\rightarrow\\ell\\nu$",
    'ZN*JetsToNuNu.*.*' : "QCD Z$\\rightarrow\\nu\\nu$",
    'EWKZ.*ZToNuNu.*' : "EWK Z$\\rightarrow\\nu\\nu$",
    'QCD.*' : "QCD",
    'Top.*' : "Top quark",
    'Diboson.*' : "WW/WZ/ZZ",
    'MET.*' : "Data"
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged input accumulator.')
    parser.add_argument('--years', nargs='*', type=int, default=[2017,2018], help='Years to run.')
    parser.add_argument('--region', default='cr_vbf_qcd', help='Name of the HF-noise enriched control region as defined in the VBF H(inv) processor.')
    parser.add_argument('--distribution', default='.*', help='Regex specifying the list of distributions to run.')
    args = parser.parse_args()
    return args

def get_hf_noise_estimate(acc, outtag, outrootfile, distribution, years=[2017, 2018], region_name='cr_vbf_qcd'):
    '''
    Calculate the noise template due to forward-jet noise (HF-noise) in VBF signal region.
    '''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    overflow = 'none'

    # Rebin if neccessary
    if distribution in binnings.keys():
        new_ax = binnings[distribution]
        h = h.rebin(new_ax.name, new_ax)
    
    elif distribution == 'dphitkpf':
        new_bins = [ibin.lo for ibin in h.identifiers('dphi') if ibin.lo < 2] + [3.5]
        
        new_ax = hist.Bin('dphi', r'$\Delta\phi_{TK,PF}$', new_bins)
        h = h.rebin('dphi', new_ax)

    outdir = f'./output/{outtag}/hf_estimate'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get data and MC yields in the QCD CR
    h = h.integrate('region', region_name)
    if distribution== 'particlenet_score':
        h = h.integrate('score_type', 'VBF-like')

    for year in years:
        # Regular expressions matching data and MC
        data = f'MET_{year}'
        mc = re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}')
        
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
            'color':'k',
            'elinewidth': 1,
        }

        fig, ax = plt.subplots()
        hist.plot1d(
            h[data], 
            overlay='dataset', 
            ax=ax, 
            overflow=overflow, 
            error_opts=data_err_opts
            )

        hist.plot1d(
            h[mc],
            overlay='dataset',
            ax=ax,
            stack=True,
            clear=False
        )

        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)
        ax.yaxis.set_ticks_position('both')

        handles, labels = ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            for regex, newlabel in legend_labels.items():
                if re.match(regex, label):
                    handle.set_label(newlabel)
                    if newlabel != 'Data':
                        handle.set_linestyle('-')
                        handle.set_edgecolor('k')
                    continue

        ax.legend(title='QCD CR', ncol=2, handles=handles)

        ax.text(0.,1.,r'QCD CR $\times$ $CR \rightarrow SR$ TF',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1.,1.,year,
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'{region_name}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

        fig, ax = plt.subplots()
        # Calculate the QCD template in SR: Data - MC in CR (already weighted by TF)
        h_qcd = h.integrate('dataset', data)        
        h_mc = h.integrate('dataset', mc)        
        h_mc.scale(-1)
        h_qcd.add(h_mc)

        hist.plot1d(h_qcd, ax=ax, overflow=overflow)
        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)
        ax.get_legend().remove()
        
        ax.text(0.,1.,'QCD Estimate in SR',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1.,1.,year,
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'qcd_estimation_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

        # Write the estimate into the output root file
        sumw = h_qcd.values(overflow=overflow)[()]
        xedges = h_qcd.axes()[0].edges(overflow=overflow)

        # Get rid of negative values
        sumw[sumw < 0.] = 0.

        outrootfile[f'hf_estimate_{distribution}_{year}'] = (sumw, xedges)

def main():
    args = parse_cli()
    inpath = args.inpath
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')
    outdir = f'./output/{outtag}/hf_estimate'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outrootpath = pjoin(outdir, 'vbfhinv_hf_estimate.root')
    outrootfile = uproot.recreate(outrootpath)
    print(f'ROOT file initiated: {outrootpath}')

    # Store the command line arguments in the INFO.txt file
    infofile = pjoin(outdir, 'INFO.txt')
    with open(infofile, 'w+') as f:
        f.write(f'QCD estimation most recently ran at: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        f.write('Command line arguments:\n\n')
        cli = vars(args)
        for arg, val in cli.items():
            f.write(f'{arg}: {val}\n')

    for distribution in distributions['sr_vbf']:
        if not re.match(args.distribution, distribution):
            continue
        get_hf_noise_estimate(acc, outtag, 
            outrootfile, 
            distribution=distribution, 
            years=args.years,
            region_name=args.region,
            )

if __name__ == '__main__':
    main()
