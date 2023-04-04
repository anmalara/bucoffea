#!/usr/bin/env python
import os, re, uproot
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from coffea import hist
from coffea.hist import poisson_interval
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio, create_legend, calculate_data_mc_ratio, set_cms_style, ratio_cosmetics

pjoin = os.path.join

# Suppress true_divide warnings
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'small',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

Bin = hist.Bin
high_pt_bins = list(range(600,1000,20))

bins = {
    'mjj':           [[50, 100., 200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500.]],
    'recoil':        [[250, 280, 310, 340, 370, 400, 430, 470, 510, 550, 590, 640, 690, 740, 790, 840, 900, 960, 1020, 1090, 1160, 1250, 1400]],
    'phi':           [50, -np.pi, np.pi],
    'eta':           [50, -5, 5],
    'frac':          [50, 0, 1],
    'mt':            [list(range(0,1000,20))],
    'met':           [list(range(0,500,50)) + list(range(500,1000,100)) + list(range(1000,2000,250))],
    'lep_pt':        [list(range(0,600,20))],
    'photon_pt':     [list(range(200,600,20)) + high_pt_bins],
    'jet_pt0':       [list(range(80,600,20)) + high_pt_bins],
    'jet_pt1':       [list(range(40,600,20)) + high_pt_bins],
    'dilepton_mass': [30,60,120],
    'dphi':          [50, 0, 3.5],
    'mult':          [10, -0.5, 9.5],
}

mjj = '$M_{jj}$ (GeV)'
pt  ='$p_{T}$ (GeV)'
eta ='$\eta$'
phi ='$\phi$'
nef = 'Neutral EM Frac'
nhf = 'Neutral Hadron Frac'
chf = 'Charged Hadron Frac'
jet0= 'Leading Jet'
jet1= 'Trailing Jet'

binnings = {
    'mjj':                Bin('mjj',             f'{mjj}',                                 *bins['mjj']),
    'particlenet_score':  Bin('score',           f'DNN score',                             *bins['frac']),
    'cnn_score':          Bin('score',           f'CNN score',                             *bins['frac']),
    'ak4_pt0':            Bin('jetpt',           f'{jet0} {pt}',                           *bins['jet_pt0']),
    'ak4_pt1':            Bin('jetpt',           f'{jet1} {pt}',                           *bins['jet_pt1']),
    'ak4_eta0':           Bin('jeteta',          f'{jet0} {eta}',                          *bins['eta']),
    'ak4_eta1':           Bin('jeteta',          f'{jet1} {eta}',                          *bins['eta']),
    'ak4_phi0':           Bin('jetphi',          f'{jet0} {phi}',                          *bins['phi']),
    'ak4_phi1':           Bin('jetphi',          f'{jet1} {phi}',                          *bins['phi']),
    'ak4_nef0':           Bin('frac',            f'{jet0} {nef}',                          *bins['frac']),
    'ak4_nef1':           Bin('frac',            f'{jet1} {nef}',                          *bins['frac']),
    'ak4_nhf0':           Bin('frac',            f'{jet0} {nhf}',                          *bins['frac']),
    'ak4_nhf1':           Bin('frac',            f'{jet1} {nhf}',                          *bins['frac']),
    'ak4_chf0':           Bin('frac',            f'{jet0} {chf}',                          *bins['frac']),
    'ak4_chf1':           Bin('frac',            f'{jet1} {chf}',                          *bins['frac']),
    'ak4_central_eta':    Bin('jeteta',          f'More Central Jet {eta}',                *bins['eta']),
    'ak4_forward_eta':    Bin('jeteta',          f'More Forward Jet {eta}',                *bins['eta']),
    'extra_ak4_mult':     Bin('multiplicity',    f'Additional Jet Multiplicity',           *bins['mult']),
    # 'dphitkpf':         Bin('dphi',            f'$\Delta\phi_{TK,PF}$',                  *bins['dphi']),
    'met':                Bin('met',             r'$p_{T}^{miss}$ (GeV)',                  *bins['met']),
    'met_phi':            Bin('phi',             r'$\phi_{MET}$',                          *bins['phi']),
    'calomet_pt':         Bin('met',             r'$p_{T,CALO}^{miss,no-\ell}$ (GeV)',     *bins['met']),
    'calomet_phi':        Bin('phi',             r'$\phi_{MET}^{CALO}$',                   *bins['phi']),
    'ak4_mult':           Bin('multiplicity',    f'AK4 multiplicity',                      *bins['mult']),
    'electron_pt':        Bin('pt',              f'Electron {pt}',                         *bins['lep_pt']),
    'electron_pt0':       Bin('pt',              f'Leading Electron {pt}',                 *bins['lep_pt']),
    'electron_pt1':       Bin('pt',              f'Trailing Electron {pt}',                *bins['lep_pt']),
    'electron_mt':        Bin('mt',              r'Electron $M_{T}$ (GeV)',                *bins['mt']),
    'muon_pt':            Bin('pt',              f'Muon {pt}',                             *bins['lep_pt']),
    'muon_pt0':           Bin('pt',              f'Leading Muon {pt}',                     *bins['lep_pt']),
    'muon_pt1':           Bin('pt',              f'Trailing Muon {pt}',                    *bins['lep_pt']),
    'muon_mt':            Bin('mt',              r'Muon $M_{T}$ (GeV)',                    *bins['mt']),
    'photon_pt0':         Bin('pt',              f'Photon {pt}',                           *bins['photon_pt']),
    'recoil':             Bin('recoil',          f'Recoil (GeV)',                          *bins['recoil']),
    'dphijr':             Bin('dphi',            f'min $\Delta\phi(j,recoil)$',            *bins['dphi']),
    'dimuon_mass':        Bin('dilepton_mass',   r'M($\mu^{+}\mu^{-}$)',                   *bins['dilepton_mass']),
    'dielectron_mass':    Bin('dilepton_mass',   r'M($e^{+}e^{-}$)',                       *bins['dilepton_mass']),
    'mjj_transformed':    Bin('transformed',     f'Rescaled {mjj}',                        *bins['eta']),
    'detajj_transformed': Bin('transformed',     r'Rescaled $\Delta\eta_{jj}$',            *bins['eta']),
    'dphijj_transformed': Bin('transformed',     r'Rescaled $\Delta\phi_{jj}$',            *bins['eta']),
}

ylims = {
    'ak4_eta0' : (1e-3,1e5),
    'ak4_eta1' : (1e-3,1e5),
    'ak4_nef0' : (1e0,1e8),
    'ak4_nef1' : (1e0,1e8),
    'ak4_nhf0' : (1e0,1e8),
    'ak4_nhf1' : (1e0,1e8),
    'ak4_chf0' : (1e0,1e8),
    'ak4_chf1' : (1e0,1e8),
    'vecb' : (1e-1,1e9),
    'vecdphi' : (1e0,1e9),
    'dphitkpf' : (1e0,1e9),
    'met' : (1e-3,1e5),
    'ak4_mult' : (1e-1,1e8),
    'particlenet_score' : (1e-1,1e5),
}

gjet = '$\\gamma$+jets'
Zll  = 'Z$\\rightarrow\\ell\\ell$'
Znn  = 'Z$\\rightarrow\\nu\\nu$'
Wln  = 'W$\\rightarrow\\ell\\nu$'
legend_labels = {
    'GJets_(DR-0p4).*':                          f'QCD {gjet}',
    '(VBFGamma|GJets_SM.*EWK).*':                f'EWK {gjet}',
    'DY.*':                                      f'QCD {Zll}',
    'EWKZ.*ZToLL.*':                             f'EWK {Zll}',
    'WN*J.*LNu.*':                               f'QCD {Wln}',
    'EWKW.*LNu.*':                               f'EWK {Wln}',
    'ZN*JetsToNuNu.*.*':                         f'QCD {Znn}',
    'EWKZ.*ZToNuNu.*':                           f'EWK {Znn}',
    'QCD.*':                                     "QCD Estimation",
    'Top.*':                                     "Top quark",
    'Diboson.*':                                 "WW/WZ/ZZ",
    'MET|Single(Electron|Photon|Muon)|EGamma.*': "Data",
    'VBF_HToInv.*':                              "VBF H(inv)",
}

legend_labels_IC = {
    'DY.*' : "$Z(\\ell\\ell)$ + jets (strong)",
    'EWKZ.*ZToLL.*' : "$Z(\\ell\\ell)$ + jets (VBF)",
    'WN*J.*LNu.*' : "$W(\\ell\\nu)$ + jets (strong)",
    'EWKW.*LNu.*' : "$W(\\ell\\nu)$ + jets (VBF)",
    'ZN*JetsToNuNu.*.*' : "$Z(\\nu\\nu)$ + jets (strong)",
    'EWKZ.*ZToNuNu.*' : "$Z(\\nu\\nu)$ + jets (VBF)",
    'QCD.*' : "QCD Estimation",
    'Top.*' : "Top quark",
    'Diboson.*' : "Dibosons",
    'MET|Single(Electron|Photon|Muon)|EGamma.*' : "Data",
    'VBF_HToInv.*' : "VBF, $B(H\\rightarrow inv)=1.0$",
}

legend_titles = {
    'sr_vbf'    : 'VBF Signal Region',
    'cr_1m_vbf' : r'VBF $1\mu$ Region',
    'cr_2m_vbf' : r'VBF $2\mu$ Region',
    'cr_1e_vbf' : r'VBF $1e$ Region',
    'cr_2e_vbf' : r'VBF $2e$ Region',
    'cr_g_vbf'  : r'VBF $\gamma$ Region',
    'sr_vbf_loose': 'VBF Loose Signal Region',
    'sr_vbf_loose_dphi': r'VBF Loose Signal Region + $\Delta\phi$',
    'sr_vbf_loose_dphi_deta': r'VBF Loose Signal Region + $\Delta\phi-\Delta\eta$',
    'cr_vbf_highdphi': r'VBF large $\Delta\phi$ Region',
    'cr_vbf_highdphi_highdeta': r'VBF large $\Delta\phi-\Delta\eta$ Region',
}

colors = {
    'DY.*' : '#ffffcc',
    'EWKW.*' : '#c6dbef',
    'EWKZ.*ZToLL.*' : '#d5bae2',
    'EWKZ.*ZToNuNu.*' : '#c4cae2',
    '.*Diboson.*' : '#4292c6',
    'Top.*' : '#6a51a3',
    '.*HF (N|n)oise.*' : '#08306b',
    '.*TT.*' : '#6a51a3',
    '.*ST.*' : '#9e9ac8',
    'ZN*JetsToNuNu.*' : '#31a354',
    'WJets.*' : '#feb24c',
    'GJets_(DR-0p4|HT).*' : '#fc4e2a',
    '(VBFGamma|GJets_SM.*EWK).*' : '#a76b51',
    'QCD.*' : '#a6bddb',
}

colors_IC = {
    'ZNJetsToNuNu_M-50_LHEFilterPtZ-FXFX.*' : (122, 189, 255),
    'EWKZ2Jets.*ZToNuNu.*' : (186, 242, 255),
    'DYJetsToLL_Pt.*FXFX.*' : (71, 191, 57),
    'EWKZ2Jets.*ZToLL.*' : (193, 255, 189),
    'WJetsToLNu_Pt.*FXFX.*' : (255, 182, 23),
    'EWKW2Jets.*WToLNu.*' : (252, 228, 159),
    'Diboson.*' : (0, 128, 128),
    'Top.*' : (148, 147, 146),
    '.*HF (N|n)oise.*' : (174, 126, 230),
}

data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

def plot_data_mc(acc, outtag, year, data, mc, data_region, mc_region, distribution='mjj', plot_signal=True, mcscale=1, binwnorm=None, fformats=['pdf'], qcd_file=None, jes_file=None, ulxs=True, is_blind=False):
    """
    Main plotter function to create a stack plot of data to background estimation (from MC).
    """
    acc.load(distribution)
    h = acc[distribution]

    # Set up overflow bin for mjj
    overflow = 'none'
    if distribution == 'mjj':
        overflow = 'over'

    # Pre-processing of the histogram, merging datasets, scaling w.r.t. XS and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h, ulxs=ulxs, mcscale=mcscale)
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

    # This sorting messes up in SR for some reason
    if data_region != 'sr_vbf':
        h.axis('dataset').sorting = 'integral'
    if distribution== 'particlenet_score':
        h = h.integrate('score_type', 'VBF-like')

    h_data = h.integrate('region', data_region)
    h_mc = h.integrate('region', mc_region)

    # Build the MC stack
    datasets = list(map(str, h[mc].identifiers('dataset')))

    plot_info = {
        'label' : [],
        'sumw' : [],
    }

    for dataset in datasets:
        sumw = h_mc.integrate('dataset', dataset).values(overflow=overflow)[()]
        if sumw.sum()<=0: continue
        plot_info['label'].append(dataset)
        plot_info['sumw'].append(sumw)

    # Get the QCD template (HF-noise estimation), only to be used in the signal region
    if 'sr_vbf' in data_region:
        qcdfilepath = pjoin(outtag,'hf_estimate','vbfhinv_hf_estimate.root')
        if qcd_file:
            qcdfilepath = qcd_file
        assert os.path.exists(qcdfilepath), f"HF-noise file cannot be found: {qcdfilepath}"
        h_qcd = uproot.open(qcdfilepath)[f'hf_estimate_{distribution}_{year}']
        # Add the HF-noise contribution (for signal region only)
        
        plot_info['label'].insert(6, 'HF Noise Estimate')
        plot_info['sumw'].insert(6, h_qcd.values * mcscale)

    fig, ax, rax = fig_ratio()
    
    # Plot data
    if not ('sr_vbf' in data_region and is_blind):
        hist.plot1d(h_data[data], ax=ax, overflow=overflow, overlay='dataset', binwnorm=binwnorm, error_opts=data_err_opts)

    xedges = h_data.integrate('dataset').axes()[0].edges(overflow=overflow)

    # Plot MC stack
    hep.histplot(plot_info['sumw'], xedges, 
        ax=ax,
        label=plot_info['label'], 
        histtype='fill',
        binwnorm=binwnorm,
        stack=True
        )

    # Plot VBF H(inv) signal if we want to
    if plot_signal:
        signal = re.compile(f'VBF_HToInvisible.*withDipoleRecoil.*{year}')

        signal_line_opts = {
            'linestyle': '-',
            'color': 'crimson',
        }

        h_signal = h.integrate('region', mc_region)[signal]
        h_signal.scale(mcscale)
        if h_signal.values(overflow=overflow)[()].sum():
            hist.plot1d(
                h_signal,
                ax=ax,
                overlay='dataset',
                overflow=overflow,
                line_opts=signal_line_opts,
                binwnorm=binwnorm,
                clear=False
                )

    ax.set_yscale('log')
    if distribution == 'mjj':
        ax.set_ylim(1e-3,1e5)
        ax.set_ylabel('Events / GeV')
    elif distribution == 'cnn_score':
        if data_region in ['cr_2m_vbf', 'cr_2e_vbf']:
            ax.set_ylim(1e-2,1e5)
        else:
            ax.set_ylim(1e-1,1e6)
    else:
        if distribution in ylims.keys():
            ax.set_ylim(ylims[distribution])
        else:
            ax.set_ylim(1e0,1e4)
        ax.set_ylabel('Events')
    
    if distribution == 'mjj':
        ax.set_xlim(left=0.)

    create_legend(ax, legend_titles.get(data_region, None), legend_labels, colors)
    set_cms_style(ax, year=year, mcscale=mcscale)

    # Plot ratio
    h_data = h_data.integrate('dataset', data)
    h_mc = h_mc.integrate('dataset', mc)

    sumw_data, sumw2_data = h_data.values(overflow=overflow, sumw2=True)[()]
    sumw_mc = h_mc.values(overflow=overflow)[()]
    
    # Add the HF-noise contribution to the background expectation
    if 'sr_vbf' in data_region:
        sumw_mc = sumw_mc + h_qcd.values * mcscale

    r, rerr = calculate_data_mc_ratio(sumw_data, sumw2_data, sumw_mc)

    # Actually do the plot if we're not blinded (only for SR)
    if not ('sr_vbf' in data_region and is_blind):
        hep.histplot(
            r,
            xedges,
            yerr=rerr,
            ax=rax,
            histtype='errorbar',
            **data_err_opts
        )

    sumw_denom, sumw2_denom = h_mc.values(overflow=overflow, sumw2=True)[()]

    unity = np.ones_like(sumw_denom)
    denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom ** 2)
    opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}
    
    rax.fill_between(
        xedges,
        np.r_[denom_unc[0], denom_unc[0, -1]],
        np.r_[denom_unc[1], denom_unc[1, -1]],
        **opts
    )

    # If a JES/JER uncertainty file is given, plot the uncertainties in the ratio pad
    if jes_file and distribution == 'mjj':
        jes_src = uproot.open(jes_file)
        h_jerUp = jes_src[f'MTR_{year}_jerUp']
        h_jerDown = jes_src[f'MTR_{year}_jerDown']
        h_jesTotalUp = jes_src[f'MTR_{year}_jesTotalUp']
        h_jesTotalDown = jes_src[f'MTR_{year}_jesTotalDown']

        # Combine JER + JES
        jecUp = 1 + np.hypot(np.abs(h_jerUp.values - 1), np.abs(h_jesTotalUp.values - 1))
        jecDown = 1 - np.hypot(np.abs(h_jerDown.values - 1), np.abs(h_jesTotalDown.values - 1))

        # Since we're looking at data/MC, take the reciprocal of these variations
        jecUp = 1/jecUp
        jecDown = 1/jecDown

        opts = {"step": "post", "facecolor": "blue", "alpha": 0.3, "linewidth": 0, "label": "JES+JER"}

        rax.fill_between(
            xedges,
            np.r_[jecUp, jecUp[-1]],
            np.r_[jecDown, jecDown[-1]],
            **opts
        )

        rax.legend()

    ratio_cosmetics(ax=rax)

    outdir = pjoin(outtag,data_region)
    os.system('mkdir -p '+outdir)
    
    # For each file format (PDF, PNG etc.), save the plot
    for fformat in fformats:
        outpath = pjoin(outdir, f'{data_region}_data_mc_{distribution}_{year}.{fformat}')
        fig.savefig(outpath)

    plt.close(fig)
