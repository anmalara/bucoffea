#!/usr/bin/env python
import os
import re
import sys
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from coffea import hist
from scipy.stats import distributions
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from bucoffea.helpers.paths import bucoffea_path
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def preprocess(h, acc, region, distribution='mjj'):
    h = merge_extensions(h, acc)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    
    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500.])
        h = h.rebin('mjj', mjj_ax)

    h = h.integrate('region', region)
    return h    

def compare_samples(acc_dict, region, distribution='mjj', years=[2017]):
    '''Compare samples from private and central Nano production.'''
    histos = {}
    for key, acc in acc_dict.items():
        acc.load(distribution)
        histos[key] = preprocess(acc[distribution], acc, region, distribution)
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    for year in years:
        datasets = [
            ('ewk_wlv', re.compile(f'EWKW2Jets.*{year}')),
            ('ewk_zvv', re.compile(f'EWKZ2Jets.*ZToNuNu.*{year}')),
        ]
        for tag, regex in datasets:
            h_private = histos['private'].integrate('dataset', regex)        
            h_central = histos['central'].integrate('dataset', regex)

            fig, ax, rax = fig_ratio()
            hist.plot1d(h_private, ax=ax, overflow='over')
            hist.plot1d(h_central, ax=ax, overflow='over', clear=False)

            ax.set_yscale('log')
            ax.set_ylim(1e-2,1e4)
            ax.legend(title='Nano', labels=['Private', 'Central'])

            ax.text(0.,1.,tag,
                fontsize=14,
                ha='left',
                va='bottom',
                transform=ax.transAxes
            )

            # Plot ratio of the two versions
            hist.plotratio(
                h_private,
                h_central,
                ax=rax,
                unc='num',
                overflow='over',
                error_opts=data_err_opts
            )

            rax.grid(True)
            rax.set_ylim(0.5,1.5)
            rax.set_ylabel('Private / Central')

            outdir = './output'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outpath = pjoin(outdir, f'private_vs_central_{tag}_{year}.pdf')
            fig.savefig(outpath)
            plt.close(fig)
            print(f'File saved: {outpath}')

def main():
    acc_dict = {
        'private' : dir_archive( bucoffea_path('submission/') ),
        'central' : dir_archive( bucoffea_path('submission/') ),
    }
    
    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw_pileup')
        acc.load('nevents')

    distributions = [
        'mjj', 
        'ak4_eta0', 
        'ak4_eta1',
        'ak4_pt0',
        'ak4_pt1',
    ]

    for distribution in distributions:
        compare_samples(acc_dict, region='sr_vbf_no_veto_all', distribution=distribution)

if __name__ == '__main__':
    main()