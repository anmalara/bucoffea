#!/usr/bin/env python
import numpy as np
from coffea import hist

Bin = hist.Bin

def obj_variables(object_name, indices, vars, extravars=None):
    if len(indices)==0:
        indices = ['']
    if extravars is not None:
        vars += extravars
    return [f"{object_name}_{var}{id}" for var in vars for id in indices]


common_distributions = [ 'mjj', 'detajj', 'dphijj', 'recoil', 'dphijr', 'particlenet_score']
common_distributions += obj_variables(object_name='ak4', indices=[0,1], vars=['eta','pt'])
# common_distributions += obj_variables(object_name='ak4', indices=[''], vars=['central_eta','forward_eta'])

# Distributions to plot for each region
distributions = {
    'sr_vbf'    :         common_distributions + obj_variables(object_name='ak4',       indices=[0,1], vars=['nef','nhf','chf']),
    'sr_vbf_nodijetcut' : common_distributions + obj_variables(object_name='ak4',       indices=[0,1], vars=['nef','nhf','chf']),
    'cr_1m_vbf' :         common_distributions + obj_variables(object_name='muon',      indices=[],    vars=['pt', 'eta', 'phi'], extravars=['mt']),
    'cr_1e_vbf' :         common_distributions + obj_variables(object_name='electron',  indices=[],    vars=['pt', 'eta', 'phi'], extravars=['mt']),
    'cr_2m_vbf' :         common_distributions + obj_variables(object_name='muon',      indices=[0,1], vars=['pt', 'eta', 'phi']) + ['dimuon_mass'],
    'cr_2e_vbf' :         common_distributions + obj_variables(object_name='electron',  indices=[0,1], vars=['pt', 'eta', 'phi']) + ['dielectron_mass'],
    'cr_g_vbf'  :         common_distributions + obj_variables(object_name='photon',    indices=[0],   vars=['pt', 'eta', 'phi']),
} 

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

binnings = {
    'particlenet_score': Bin('score', r'DNN score', 50, 0, 1),
    'mjj': Bin('mjj', r'$M_{jj} \ (GeV)$', [50, 100., 200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]),
    'ak4_pt0': Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1': Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) ),
    'ak4_phi0' : Bin("jetphi", r"Leading AK4 jet $\phi$", 50,-np.pi, np.pi),
    'ak4_phi1' : Bin("jetphi", r"Trailing AK4 jet $\phi$", 50,-np.pi, np.pi),
    'ak4_eta0' : Bin("jeteta", r"Leading Jet $\eta$", 50, -5, 5),
    'ak4_eta1' : Bin("jeteta", r"Leading Jet $\eta$", 50, -5, 5),
    'ak4_central_eta' : Bin("jeteta", r"More Central Jet $\eta$", 50, -5, 5),
    'ak4_forward_eta' : Bin("jeteta", r"More Forward Jet $\eta$", 50, -5, 5),
    'ak4_nef0' : Bin('frac', 'Leading Jet Neutral EM Frac', 50, 0, 1),
    'ak4_nef1' : Bin('frac', 'Trailing Jet Neutral EM Frac', 50, 0, 1),
    'ak4_nhf0' : Bin('frac', 'Leading Jet Neutral Hadronic Frac', 50, 0, 1),
    'ak4_nhf1' : Bin('frac', 'Trailing Jet Neutral Hadronic Frac', 50, 0, 1),
    'ak4_chf0' : Bin('frac', 'Leading Jet Charged Hadronic Frac', 50, 0, 1),
    'ak4_chf1' : Bin('frac', 'Trailing Jet Charged Hadronic Frac', 50, 0, 1),
    # 'dphitkpf' : Bin('dphi', r'$\Delta\phi_{TK,PF}$', 50, 0, 3.5),
    'met' : Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,500,50)) + list(range(500,1000,100)) + list(range(1000,2000,250))),
    'met_phi' : Bin("phi", r"$\phi_{MET}$", 50, -np.pi, np.pi),
    'ak4_mult' : Bin("multiplicity", r"AK4 multiplicity", 10, -0.5, 9.5),
    'ak4_mt0' : Bin("mt", r"Leading AK4 $M_{T}$ (GeV)", 50, 0, 1000),
    'ak4_mt1' : Bin("mt", r"Trailing AK4 $M_{T}$ (GeV)", 50, 0, 1000),
    'dphijr' : Bin("dphi", r"$min\Delta\phi(j,recoil)$", 50, 0, 3.5),
    'extra_ak4_mult' : Bin("multiplicity", r"Additional AK4 Jet Multiplicity", 10, -0.5, 9.5),
    'recoil' : Bin('recoil','Recoil (GeV)', recoil_bins_2016),
}