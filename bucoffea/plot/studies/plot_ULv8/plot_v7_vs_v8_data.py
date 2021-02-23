#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from scipy.stats import distributions

from bucoffea.plot.util import merge_datasets, merge_extensions, fig_ratio
from bucoffea.plot.style import matplotlib_rc
from coffea import hist
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive

pjoin = os.path.join

matplotlib_rc()

def preprocess(h, acc, distribution, region='sr_vbf', dataset='MET_2017'):
    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    h = h.integrate('region', region).integrate('dataset', dataset)

    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)

    return h

def plot_v7_vs_v8_data(acc_v7, acc_v8, distribution, region='sr_vbf'):
    '''Plot the comparison of v7 and v8 MET datasets (2017 for now)'''
    acc_v7.load(distribution)
    acc_v8.load(distribution)

    h_v7 = preprocess(acc_v7[distribution], acc_v7, distribution, region)
    h_v8 = preprocess(acc_v8[distribution], acc_v8, distribution, region)

    fig, ax, rax = fig_ratio()
    hist.plot1d(h_v7, ax=ax)
    hist.plot1d(h_v8, ax=ax, clear=False)

    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e5)

    ax.legend(title='MET 2017',labels=['Nano v7', 'Nano v8'])

    if region == 'sr_vbf':
        regiontext = 'VBF Signal Region'
    elif region == 'sr_vbf_without_mitigation_cuts':
        regiontext = 'VBF Signal Region (No mitigation cuts)'

    ax.text(0.,1.,regiontext,
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    # Plot v8 / v7 ratio
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    hist.plotratio(h_v8, h_v7, 
            ax=rax,
            unc='num',
            error_opts=data_err_opts
            )

    newxlabels = {
        'ak4_eta0' : r'Leading jet $\eta$',
        'ak4_eta1' : r'Trailing jet $\eta$',
        'detajj' : r'$\Delta\eta_{jj}$',
    }

    if distribution in newxlabels.keys():
        ax.set_xlabel(newxlabels[distribution])
        rax.set_xlabel(newxlabels[distribution])

    rax.set_ylabel('v8 / v7')
    rax.grid(True)
    if distribution in ['mjj','detajj']:
        rax.set_ylim(0.6,1.4)
        loc = MultipleLocator(0.2)
        rax.yaxis.set_major_locator(loc)
    else:
        rax.set_ylim(0,2)
        loc = MultipleLocator(0.5)
        rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = './output/v7_vs_v8'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'met_2017_v7_v8_{region}_{distribution}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath_v7 = '../../../submission/merged_2021-02-18_vbfhinv_03Sep20v7_MET_2017'
    inpath_v8 = '../../../submission/merged_2021-02-18_vbfhinv_ULv8_MET_2017'
    acc_v7 = dir_archive(inpath_v7) 
    acc_v8 = dir_archive(inpath_v8) 

    for acc in [acc_v7, acc_v8]:
        acc.load('sumw')
        acc.load('sumw2')

    distributions = ['mjj', 'detajj', 'ak4_eta0', 'ak4_eta1']

    regions = ['sr_vbf', 'sr_vbf_without_mitigation_cuts']

    for region in regions:
        for distribution in distributions:
            plot_v7_vs_v8_data(acc_v7, acc_v8, distribution=distribution, region=region)

if __name__ == '__main__':
    main()