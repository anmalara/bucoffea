#!/usr/bin/env python

import os, re
from klepto.archives import dir_archive
import mplhep as hep
from matplotlib import pyplot as plt
import numpy as np
from bucoffea.plot.plotter import binnings, legend_titles, legend_labels
from bucoffea.plot.util import merge_datasets_and_scale, rebin, fig_ratio, create_legend, ratio_cosmetics, ratio_unc
from coffea import hist
from coffea.hist import poisson_interval

pjoin = os.path.join

colors = {
    'sr_vbf_no_veto_all' : 'k',
    'sr_vbf_nodijetcut' : '#31a354', #green
    'cr_1m_vbf' : '#2e74db', #blue
    'cr_1e_vbf' : '#e66b0e', #orange
    'cr_2m_vbf' : '#6a51a3', #violet
    'cr_2e_vbf' : '#ad020a', #red
    'cr_g_vbf' : '#4c9ea8', #water
    'sr_vbf_loose': '#3e4042',
    'sr_vbf_loose_dphi': '#5b5c5e',
    'sr_vbf_loose_deta': '#a7a9ab',
    'sr_vbf_loose_dphi_deta': '#9ba3ab',
    'sr_vbf_highdphi': '#ebbdb5',
    'sr_vbf_highdphi_mjj': '#e6a79c',
    'sr_vbf_highdphi_highdeta': '#db442a',
}

def get_regions(dataset):
    matching = {
            'sr_vbf_no_veto_all' : re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            # 'sr_vbf_nodijetcut' : re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'cr_1e_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*|EGamma_201*.*).*'),
            'cr_1m_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'cr_2m_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX|MET_201*.*).*'),
            'cr_2e_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX|EGamma_201*.*).*'),
            'cr_g_vbf' : re.compile(f'(GJets_DR-0p4.*|VBFGamma.*|QCD_data.*|EGamma_201*.*).*'),
            'sr_vbf_loose': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_loose_dphi': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_loose_deta': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_loose_dphi_deta': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_highdphi': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_highdphi_mjj': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
            'sr_vbf_highdphi_highdeta': re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*|MET_201*.*).*'),
        }
    return [x for x in matching if re.match(matching[x], dataset)]

def shape_comparison(inpath, distribution, year, region_ref_ = 'sr_vbf_no_veto_all'):
    outdir = pjoin('./output/',inpath.replace('..','').replace('/',''),'shape_comparison_2')
    os.system('mkdir -p '+outdir)

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')
    acc.load(distribution)

    histograms = acc[distribution]
    if distribution== 'particlenet_score':
        histograms = histograms.integrate('score_type', 'VBF-like')
    histograms = merge_datasets_and_scale(histograms, acc, reweight_pu=False, noscale=False)
    
    #TODO sumw_pileup
    histograms = rebin(histograms, distribution, binnings)
    datasets = list(map(str, histograms.identifiers('dataset')))
    for dataset in datasets:
        regions = get_regions(dataset)
        if len(regions)==0: continue
        region_ref = region_ref_ if region_ref_ in regions else regions[0]
        dataset_label = [label for regex, label in legend_labels.items() if re.match(regex, dataset)][0]
        if dataset_label=='Data':
            # regions = list(filter(lambda x: not 'sr' in x, regions))
            # region_ref = regions[0]
            dataset_label = dataset.replace('_'+year, '')
        dataset_name = dataset_label.replace(' ', '_').replace('/', '_').replace('+', '').replace('$', '').replace('\\', '').replace('rightarrow', '')
        histogram = histograms.integrate('dataset', dataset)
        print(dataset, regions)
        histogram.scale({k[0]:1./histogram.values()[k].sum() for k in histogram.values()}, axis='region')
        
        fig, ax, rax = fig_ratio()
        xedges = histogram[region_ref].axes()[1].edges()

        den, den_err = histogram[region_ref].values(sumw2=True)[(region_ref,)]
        den_err = np.sqrt(den_err)
        for region in regions:
            region_label = [label for regex, label in legend_titles.items() if re.match(regex, region)][0]
            num, num_err = histogram[region].values(sumw2=True)[(region,)]
            num_err = np.sqrt(num_err)
            r = num/den
            rerr = ratio_unc(den, num, den_err, num_err)
            r[np.isnan(r) | np.isinf(r)] = 0.
            rerr[np.isnan(rerr) | np.isinf(rerr)] = 0.
            hep.histplot(num, xedges, yerr=num_err, ax=ax, histtype='errorbar', label=region_label, color=colors[region])
            hep.histplot(r, xedges, yerr=rerr, ax=rax, histtype='errorbar', color=colors[region])
        
        create_legend(ax, dataset_label, legend_titles)
        ratio_cosmetics(ax=rax, yaxis ='ratio', ylims=(0.,2.0), ystep=0.5)
        outpath = pjoin(outdir, f'{distribution}_{dataset_name}_{year}.pdf')
        fig.savefig(outpath)


def main():
    inpath = '../merged_files/PFNANO_V9_17Feb23_PostNanoTools_2'
    distribution = 'particlenet_score'
    year = '2018'
    shape_comparison(inpath, distribution, year)


if __name__ == '__main__':
    main()