#!/usr/bin/env python
import argparse
import warnings
import os
import re

from klepto.archives import dir_archive
from pprint import pprint
from distributions import distributions
from tqdm import tqdm

from bucoffea.plot.plotter import plot_data_mc
from bucoffea.plot.util import get_mc_scales, dump_info

pjoin = os.path.join

warnings.filterwarnings('ignore', category=RuntimeWarning)

def make_plot(args):
    acc = dir_archive(args.inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = pjoin('./output/',args.inpath.replace('..','').replace('/',''))
    dump_info(args, outtag)

    for year in args.years:
        data = {
            'sr_vbf' : f'MET_{year}',
            'sr_vbf_nodijetcut' : f'MET_{year}',
            'cr_1m_vbf' : f'MET_{year}',
            'cr_2m_vbf' : f'MET_{year}',
            'cr_1e_vbf' : f'EGamma_{year}',
            'cr_2e_vbf' : f'EGamma_{year}',
            'cr_g_vbf'  : f'EGamma_{year}',
        }

        mc = {
            'sr_vbf_no_veto_all' : re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
            'sr_vbf_nodijetcut' : re.compile(f'(ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
            'cr_1m_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
            'cr_1e_vbf' : re.compile(f'(EWKW.*|EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt_FXFX.*|WJetsToLNu_Pt-FXFX.*).*{year}'),
            'cr_2m_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX).*{year}'),
            'cr_2e_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|DYJetsToLL.*Pt.*FXFX).*{year}'),
            'cr_g_vbf' : re.compile(f'(GJets_DR-0p4.*|VBFGamma.*|QCD_data.*).*{year}'),
        }

        # Dictionary having the region -> scale factor mapping, by default this is an empty dict
        mc_scales = get_mc_scales(args.mcscales)

        for data_region in data.keys():
            if not re.match(args.region, data_region):
                continue

            # Pick the signal region with veto weights (instead of hard-lepton veto)
            # for the VBF signal region
            if data_region == 'sr_vbf':
                mc_region = 'sr_vbf_no_veto_all'
            else:
                mc_region = data_region

            # Figure out the MC scaling to apply for this region, default is 1
            mcscale = mc_scales.get(mc_region, 1)
            if mcscale != 1:
                print(f'INFO: Region {mc_region}')
                print(f'INFO: MC scale {mcscale}')
            
            # If we want to do 1/5th unblinding on the data
            if args.one_fifth_unblind and data_region == 'sr_vbf':
                mcscale *= 0.2

            for distribution in tqdm(distributions[data_region], desc=f"Plotting for {data_region}"):
                if not re.match(args.distribution, distribution):
                    continue
                
                try:
                    plot_data_mc(acc, outtag, year,
                        data=data[data_region],
                        mc=mc[mc_region],
                        data_region=data_region,
                        mc_region=mc_region,
                        distribution=distribution,
                        mcscale=mcscale,
                        plot_signal='sr_vbf' in data_region,
                        jes_file='./jec/jes_uncs.root' if args.jes else None,
                        ulxs=not args.eoyxs,
                        fformats=args.fformats,
                        binwnorm=1 if distribution == 'mjj' else None,
                        is_blind=args.blind,
                    )
                except KeyError as e:
                    print(str(e))
                    print(f'WARNING: {data_region} not found in inputs, skipping.')
                    continue

def commandline():
    parser = argparse.ArgumentParser(prog='Plotter.')
    parser.add_argument('inpath', type=str, help='Input folder to use.')
    parser.add_argument('--region', type=str, default='.*', help='Regex specifying the analysis regions to plot.')
    parser.add_argument('--distribution', type=str, default='.*', help='Regex specifying the distributions to plot.')
    parser.add_argument('--years', type=int, nargs='*', default=[2017,2018], help='Years to run on.')
    parser.add_argument('--one_fifth_unblind', action='store_true', help='1/5th unblinded data.')
    parser.add_argument('--blind', action='store_true', help='blind data.')
    parser.add_argument('--fformats', nargs='*', default=['pdf'], help='Output file format for the plots, default is PDF only.')
    parser.add_argument('--jes', action='store_true', help='Plot JES+JER uncertainty bands.')
    parser.add_argument('--eoyxs', action='store_true', help='Use EOY XS for normalization, otherwise use UL XS.')
    parser.add_argument('--mcscales', type=str, default=None, help='An optional txt file for storing the MC scales per region.')
    args = parser.parse_args()
    return args

def main():
    args = commandline()
    make_plot(args)

if __name__ == "__main__":
    main()
