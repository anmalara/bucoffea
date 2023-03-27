#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi, dump_info, create_legend, set_cms_style
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint
from distributions import distributions
from bucoffea.plot.plotter import binnings, legend_labels, colors

pjoin = os.path.join
Bin = hist.Bin

def get_hf_noise_estimate(acc, outdir, outrootfile, distribution, years=[2017, 2018], region_name='cr_vbf_qcd'):
    '''
    Calculate the noise template due to forward-jet noise (HF-noise) in VBF signal region.
    '''
    acc.load(distribution)
    h = acc[distribution]

    # Set up overflow bin for mjj
    overflow = 'none'
    if distribution == 'mjj':
        overflow = 'over'

    # Pre-processing of the histogram, merging datasets, scaling w.r.t. XS and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin if necessary
        # Specifically rebin dphitkpf distribution: Merge the bins in the tails
    # Annoying approach but it works (due to float precision problems)
    if distribution in binnings.keys():
        new_ax = binnings[distribution]
        h = h.rebin(new_ax.name, new_ax)
    elif distribution == 'dphitkpf':
        new_bins = [ibin.lo for ibin in h.identifiers('dphi') if ibin.lo < 2] + [3.5]
        new_ax = Bin('dphi', r'$\Delta\phi_{TK,PF}$', new_bins)
        h = h.rebin('dphi', new_ax)

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

        create_legend(ax, legend_title='QCD CR', legend_labels=legend_labels, colors=colors)
        set_cms_style(ax, year=year, extratext=r'QCD CR $\times$ $CR \rightarrow SR$ TF', size = 0.75)

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
        set_cms_style(ax, year=year, extratext='QCD Estimate in SR')
        
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

def commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to the merged input accumulator.')
    parser.add_argument('--years', nargs='*', type=int, default=[2017,2018], help='Years to run.')
    parser.add_argument('--region', default='cr_vbf_qcd', help='Name of the HF-noise enriched control region as defined in the VBF H(inv) processor.')
    parser.add_argument('--distribution', default='.*', help='Regex specifying the list of distributions to run.')
    args = parser.parse_args()
    return args

def main():
    args = commandline()
    inpath = args.inpath
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outdir = pjoin('./output/',args.inpath.replace('..','').replace('/',''),'hf_estimate')
    dump_info(args, outdir)

    outrootpath = pjoin(outdir, 'vbfhinv_hf_estimate.root')
    outrootfile = uproot.recreate(outrootpath)
    print(f'ROOT file initiated: {outrootpath}')

    for distribution in distributions['sr_vbf']:
        if not re.match(args.distribution, distribution):
            continue
        get_hf_noise_estimate(acc, outdir,
            outrootfile, 
            distribution=distribution, 
            years=args.years,
            region_name=args.region,
            )

if __name__ == '__main__':
    main()
