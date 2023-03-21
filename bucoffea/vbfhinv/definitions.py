import copy
import re
from coffea import hist

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from awkward import JaggedArray
import numpy as np
from bucoffea.helpers import object_overlap, sigmoid3, dphi
from bucoffea.helpers.dataset import extract_year
from bucoffea.helpers.paths import bucoffea_path
from bucoffea.helpers.gen import find_first_parent
from bucoffea.monojet.definitions import accu_int, defaultdict_accumulator_of_empty_column_accumulator_float16, defaultdict_accumulator_of_empty_column_accumulator_int64,defaultdict_accumulator_of_empty_column_accumulator_bool, defaultdict_accumulator_of_empty_column_accumulator_ndarray_uint8
from pprint import pprint

def vbfhinv_accumulator(cfg):
    dataset_ax = Cat("dataset", "Primary dataset")
    unc_ax = Cat("uncertainty", "Uncertainty weight variation")
    variation_ax = Cat("variation", "Uncertainty weight variation")
    region_ax = Cat("region", "Selection region")
    type_ax = Cat("type", "Type")
    
    # Score type for ParticleNet (i.e. ggH vs VBF)
    score_type_ax = Cat("score_type", "Score type")

    vpt_ax = Bin("vpt",r"$p_{T}^{V}$ (GeV)", 100, 0, 2000)

    met_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 200, 0, 2000)
    recoil_ax = Bin("recoil", r"Recoil (GeV)", 200, 0, 2000)

    score_ax = Bin("score", "Neural network score", 100, 0, 1)
    transformed_ax = Bin("transformed", "Transformed value", 50, -5, 5)

    mjj_ax = Bin("mjj", r"$M_{jj}$ (GeV)", 150, 0, 7500)
    mjj_ax_coarse = Bin("mjj", r"$M_{jj}$ (GeV)", [200,500,1000,1500,2000,3000,7500])
    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    jet_pt_ax_coarse = Bin("jetpt", r"$p_{T}$ (GeV)", 5, 0, 500)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)
    jet_eta_ax_coarse = Bin("jeteta", r"$\eta$", 10, -5, 5)
    jet_phi_ax = Bin("jetphi", r"$\phi$", 50,-np.pi, np.pi)

    jet_eta_ax_very_coarse = Bin("jeteta", r"$\eta$", [0, 2.5, 3, 3.25, 5])

    jet_mass_ax = Bin("mass", r"$M_{jet}$ (GeV)", 100,0,300)

    dpfcalo_ax = Bin("dpfcalo", r"$(PFMET-CaloMET) / Recoil$", 20, -1, 1)
    dpftk_ax = Bin("dpftk", r"$(PFMET-TkMET) / MET$", 20, -1, 1)
    btag_ax = Bin("btag", r"B tag discriminator", 20, 0, 1)
    multiplicity_ax = Bin("multiplicity", r"multiplicity", 10, -0.5, 9.5)
    nconst_ax = Bin("nconst", r"Number of constituents", 25, -0.5, 99.5)
    dphi_ax = Bin("dphi", r"$\Delta\phi$", 50, 0, 3.5)
    deta_ax = Bin("deta", r"$\Delta\eta$", 50, 0, 10)
    dr_ax = Bin("dr", r"$\Delta R$", 50, 0, 2)

    pt_ax = Bin("pt", r"$p_{T}$ (GeV)", 100, 0, 1000)
    pt_ax_mu = Bin("pt", r"$p_{T}$ (GeV)", [20,25,30,40,50,60,120])
    pt_ax_el = Bin("pt", r"$p_{T}$ (GeV)", [10,20,35,50,100,200,500])
    pt_ax_tau = Bin("pt", r"$p_{T}$ (GeV)", [18,20,25,30,35,40,500,1000])

    dphitkpf_ax_coarse = Bin("dphi", r"$\Delta\phi$", [0, 0.5, 1, 1.5, 2, np.pi])
    jet_abseta_ax_coarse = Bin("jeteta", r"Jet $|\eta|$", [0, 2.5, 3, 3.25, 5])

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 100, 0, 4000)
    mt_ax = Bin("mt", r"$M_{T}$ (GeV)", 100, 0, 1000)
    eta_ax = Bin("eta", r"$\eta$", 50, -5, 5)
    eta_ax_el = Bin("eta", r"$\eta$", [-2.5, -2.0, -1.56, -1.44, -0.8, 0, 0.8, 1.44,1.56,2.0,2.5])
    abseta_ax_mu = Bin("abseta", r"$|\eta|$", [0,0.9,1.2,2.1,2.4])

    eta_ax_coarse = Bin("eta", r"$\eta$", 25, -5, 5)
    phi_ax = Bin("phi", r"$\phi$", 50,-np.pi, np.pi)
    phi_ax_coarse = Bin("phi", r"$\phi$", 20,-np.pi, np.pi)

    tau21_ax = Bin("tau21", r"Tagger", 50,-5,5)
    tagger_ax = Bin("tagger", r"Tagger", 50,-5,5)

    dilepton_mass_ax = Bin("dilepton_mass", r"$M(\ell\ell)$ (GeV)", 100,50,150)

    sigma_eta_eta_ax = Bin("sigmaetaeta", r"$\sigma_{\eta\eta}$", 50, 0, 0.5)
    sigma_phi_phi_ax = Bin("sigmaphiphi", r"$\sigma_{\phi\phi}$", 50, 0, 0.5)
    central_eta_stripsize_ax = Bin("centraletastripsize", r"HF central $\eta$ Strip Size", 5, -0.5, 4.5)
    adjacent_eta_stripsize_ax = Bin("adjacentetastripsize", r"HF adjacent $\eta$ Strip Size", 5, -0.5, 4.5)
    eta_hf_ax = Bin("jeta", r"Jet $|\eta|$", [2.9, 3.25, 5])
    ak40_abseta_ax = Bin("jeta", r"Jet $|\eta|$", [0, 2.5, 2.9, 5])

    sigma_eta_phi_diff_ax = Bin("sigmaetaminusphi", r"$\sigma_{\eta\eta} - \sigma_{\phi\phi}$", 30, -0.3, 0.3)

    vecb_ax = Bin("vecb", r"VecB", 50, 0, 1)
    vecdphi_ax = Bin("vecdphi", r"VecDPhi", 60, 0, 3)

    weight_type_ax = Cat("weight_type", "Weight type")
    weight_ax = Bin("weight_value", "Weight",100,0.5,1.5)
    weight_wide_ax = Bin("weight_value", "Weight",100,-10,10)

    nvtx_ax = Bin('nvtx','Number of vertices',100,-0.5,99.5)
    rho_ax = Bin('rho','Energy density',100, 0, 100)
    frac_ax = Bin('frac','Fraction', 50, 0, 1)
    Hist = hist.Hist
    items = {}
    items["genvpt_check"] = Hist("Counts", dataset_ax, type_ax, vpt_ax)

    items["gen_vpt"] = Hist("Counts", dataset_ax, vpt_ax, region_ax)
    items["gen_mjj"] = Hist("Counts", dataset_ax, mjj_ax, region_ax)
    items["lhe_njets"] = Hist("Counts", dataset_ax, multiplicity_ax)
    items["lhe_ht"] = Hist("Counts", dataset_ax, ht_ax)
    items["lhe_htinc"] = Hist("Counts", dataset_ax, ht_ax)
    items["calomet_pt"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["calomet_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["met"] = Hist("Counts", dataset_ax, region_ax, met_ax)
    items["met_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["recoil"] = Hist("Counts", dataset_ax, region_ax, recoil_ax)
    items["recoil_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)

    # ParticleNet score distribution
    items["particlenet_score"] = Hist("Counts", dataset_ax, region_ax, score_ax, score_type_ax)

    items["mjj"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
    items["mjj_noewk"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
    items["mjj_nopref"] = Hist("Counts", dataset_ax, region_ax, mjj_ax)
    items["mjj_veto_weight"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_ele_trig_weight"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_ele_id"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_ele_reco"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_muon_id"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_muon_iso"] = Hist("Counts", dataset_ax, region_ax, variation_ax, mjj_ax)
    items["mjj_unc"] = Hist("Counts", dataset_ax, region_ax, mjj_ax, unc_ax)
    items["dphijj"] = Hist("Counts", dataset_ax, region_ax, dphi_ax)
    items["detajj"] = Hist("Counts", dataset_ax, region_ax, deta_ax)

    items["mjj_ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, mjj_ax_coarse, jet_eta_ax)

    items["ak4_pt0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_ptraw0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi0"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_chf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nef0"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nconst0"] = Hist("Counts", dataset_ax, region_ax, nconst_ax)
    items["ak4_mt0"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

    items["ak4_pt1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_ptraw1"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta1"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi1"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_chf1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nhf1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nef1"] = Hist("Counts", dataset_ax, region_ax, frac_ax)
    items["ak4_nconst1"] = Hist("Counts", dataset_ax, region_ax, nconst_ax)
    items["ak4_mt1"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

    items["ak4_central_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_forward_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)

    # Leading jet pt vs eta
    items["ak4_pt0_eta0"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax,jet_eta_ax_coarse)

    items["ak4_pt"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_pt_nopref"] = Hist("Counts", dataset_ax, region_ax, jet_pt_ax)
    items["ak4_eta_nopref"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax)
    items["ak4_phi_nopref"] = Hist("Counts", dataset_ax, region_ax, jet_phi_ax)
    items["ak4_btag"] = Hist("Counts", dataset_ax, region_ax, btag_ax)

    items["ak4_eta_phi"] = Hist("Counts", dataset_ax, region_ax, jet_eta_ax, jet_phi_ax)

    items["photon_eta_phi"] = Hist("Counts", dataset_ax, region_ax, eta_ax_coarse, phi_ax_coarse)

    items["dpfcalo_cr"] = Hist("Counts", dataset_ax, region_ax, dpfcalo_ax)
    items["dpfcalo_sr"] = Hist("Counts", dataset_ax, region_ax, dpfcalo_ax)
    items["dphijm"] = Hist("min(4 leading jets, MET)", dataset_ax, region_ax, dphi_ax)
    items["dphijr"] = Hist("min(4 leading jets, Recoil)", dataset_ax, region_ax, dphi_ax)

    items["vecb"] = Hist("Counts", dataset_ax, region_ax, vecb_ax)
    items["vecdphi"] = Hist("Counts", dataset_ax, region_ax, vecdphi_ax)
    items["dphitkpf"] = Hist("Counts", dataset_ax, region_ax, dphi_ax)

    # Multiplicity histograms
    for cand in ['ak4', 'ak8', 'bjet', 'loose_ele', 'loose_muo', 'tight_ele', 'tight_muo', 'tau', 'photon','hlt_single_muon','muons_hltmatch']:
        items[f"{cand}_mult"] = Hist(cand, dataset_ax, region_ax, multiplicity_ax)

    items["extra_ak4_mult"] = Hist(cand, dataset_ax, region_ax, multiplicity_ax)

    if cfg.RUN.SAVE_HF_VARIABLES:
        items["ak4_sigma_eta_eta0"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax)
        items["ak4_sigma_phi_phi0"] = Hist("Counts", dataset_ax, region_ax, sigma_phi_phi_ax)
        items["ak4_sigma_eta_eta1"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax)
        items["ak4_sigma_phi_phi1"] = Hist("Counts", dataset_ax, region_ax, sigma_phi_phi_ax)
        items["ak4_sigma_eta_eta"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax, eta_hf_ax)
        items["ak4_sigma_phi_phi"] = Hist("Counts", dataset_ax, region_ax, sigma_phi_phi_ax, eta_hf_ax)

        # Two dimensional
        items["ak4_sigma_eta_phi"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax, sigma_phi_phi_ax, eta_hf_ax)
        items["ak4_sigma_eta_phi0"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax, sigma_phi_phi_ax, eta_hf_ax)
        items["ak4_sigma_eta_phi1"] = Hist("Counts", dataset_ax, region_ax, sigma_eta_eta_ax, sigma_phi_phi_ax, eta_hf_ax)
        items["ak4_hfcentral_adjacent_etastripsize"] = Hist("Counts", dataset_ax, region_ax, central_eta_stripsize_ax, adjacent_eta_stripsize_ax, eta_hf_ax)
        items["ak4_hfcentral_adjacent_etastripsize0"] = Hist("Counts", dataset_ax, region_ax, central_eta_stripsize_ax, adjacent_eta_stripsize_ax, eta_hf_ax)
        items["ak4_hfcentral_adjacent_etastripsize1"] = Hist("Counts", dataset_ax, region_ax, central_eta_stripsize_ax, adjacent_eta_stripsize_ax, eta_hf_ax)

    items["muon_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_pt_abseta"] = Hist("Counts", dataset_ax, region_ax, pt_ax_mu, abseta_ax_mu)
    items["muon_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["muon_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["muon_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["muon_mt"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

    items["dimuon_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["dimuon_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["dimuon_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)

    items["electron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_pt_eta"] = Hist("Counts", dataset_ax, region_ax, pt_ax_el, eta_ax_el)
    items["electron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_pt0"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_eta0"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi0"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_pt1"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["electron_eta1"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["electron_phi1"] = Hist("Counts", dataset_ax, region_ax, phi_ax)
    items["electron_mt"] = Hist("Counts", dataset_ax, region_ax, mt_ax)

    items["dielectron_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items["dielectron_eta"] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items["dielectron_mass"] = Hist("Counts", dataset_ax, region_ax, dilepton_mass_ax)

    items['photon_pt0'] = Hist("Counts", dataset_ax, region_ax, pt_ax)
    items['photon_eta0'] = Hist("Counts", dataset_ax, region_ax, eta_ax)
    items['photon_phi0'] = Hist("Counts", dataset_ax, region_ax, phi_ax)

    items["tau_pt"] = Hist("Counts", dataset_ax, region_ax, pt_ax_tau)

    # One cutflow counter per region
    regions = vbfhinv_regions(cfg).keys()
    for region in regions:
        if region=="inclusive":
            continue
        items[f'cutflow_{region}']  = processor.defaultdict_accumulator(accu_int)

    items['nevents'] = processor.defaultdict_accumulator(float)
    items['sumw'] = processor.defaultdict_accumulator(float)
    items['sumw2'] = processor.defaultdict_accumulator(float)
    items['sumw_pileup'] = processor.defaultdict_accumulator(float)

    items['selected_events'] = processor.defaultdict_accumulator(list)
    items['kinematics'] = processor.defaultdict_accumulator(list)

    items['weights'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_ax)
    items['weights_wide'] = Hist("Weights", dataset_ax, region_ax, weight_type_ax, weight_wide_ax)
    items['npv'] = Hist('Number of primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npvgood'] = Hist('Number of good primary vertices', dataset_ax, region_ax, nvtx_ax)
    items['npv_nopu'] = Hist('Number of primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)
    items['npvgood_nopu'] = Hist('Number of good primary vertices (No PU weights)', dataset_ax, region_ax, nvtx_ax)

    items['rho_all'] = Hist(r'$\rho$ for all PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_central'] = Hist(r'$\rho$ for central PF candidates', dataset_ax, region_ax, rho_ax)
    items['rho_all_nopu'] = Hist(r'$\rho$ for all PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)
    items['rho_central_nopu'] = Hist(r'$\rho$ for central PF candidates (No PU weights)', dataset_ax, region_ax, rho_ax)

    items['tree_float16'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_float16)
    items['tree_int64'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_int64)
    items['tree_bool'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_bool)
    items['tree_ndarray'] = processor.defaultdict_accumulator(defaultdict_accumulator_of_empty_column_accumulator_ndarray_uint8)
    return  processor.dict_accumulator(items)

def vbfhinv_regions(cfg):
    # 'inclusive'    # 'veto_b',
    def remove_items(orig_list, to_remove):
        return list(filter(lambda x: x not in to_remove, orig_list))
    
    def append_items(orig_list, to_add):
        return list(orig_list+ to_add)
    
    common_cuts = [
        'filt_met',
        'veto_ele',
        'veto_muo',
        'veto_photon',
        'veto_tau',
        'veto_b',
        'mindphijr',
        'recoil',
        'two_jets',
        'leadak4_pt_eta',
        'trailak4_pt_eta',
        'leadak4_id',
        'trailak4_id',
        'hemisphere',
        'mjj',
        'dphijj',
        'detajj',
        'leadak4_clean'
    ]

    if cfg.RUN.APPLY_HF_CUTS:
        common_cuts.extend([
            'central_stripsize_cut',
            'sigma_eta_minus_phi'
        ])

    # The regular ReReco cleaning cuts
    if cfg.RUN.APPLY_CLEANING_CUTS:
        common_cuts.extend([
            'max_neEmEF',
            'veto_hfhf',
            # 'leadak4_not_in_hf',
        ])
    
    # mjj > 1.2 TeV
    if cfg.RUN.TIGHT_MJJ_CUT:
        common_cuts.append('mjj_tight')

    regions = {}
    # regions['inclusive'] = ['inclusive']

    # Signal regions (v = mono-V, j = mono-jet)
    regions['sr_vbf'] = ['trig_met','metphihemextveto','hornveto'] + common_cuts + ['dpfcalo_sr', 'eemitigation']

    if cfg.RUN.ONE_FIFTH_UNBLIND:
        regions['sr_vbf'].insert(0, 'one_fifth_mask')

    if not cfg.RUN.APPLY_CLEANING_CUTS:
        regions['sr_vbf'].remove('hornveto')
        regions['sr_vbf'].remove('eemitigation')

    if cfg.RUN.REGION_WITHOUT_DIJET_CUTS:
        regions['sr_vbf_nodijetcut'] = remove_items(regions['sr_vbf'], ['mjj','detajj','dphijj'])

    # SR without PU weights
    # regions['sr_vbf_no_pu'] = copy.deepcopy(regions['sr_vbf'])


    # SR without HEM veto
    if cfg.RUN.HEMCHECK:
        regions['sr_vbf_no_hem_veto'] = remove_items(regions['sr_vbf'], ['metphihemextveto'])

    # QCD CR with the HF shape cuts inverted
    if cfg.RUN.QCD_ESTIMATION:
        to_remove = ['central_stripsize_cut', 'sigma_eta_minus_phi']
        regions['cr_vbf_qcd'] = remove_items(regions['sr_vbf'], to_remove)
        if 'one_fifth_mask' in regions['cr_vbf_qcd']:
            regions['cr_vbf_qcd'].remove('one_fifth_mask')
        regions['cr_vbf_qcd'].append('fail_hf_cuts')
        if cfg.RUN.REGION_WITHOUT_DIJET_CUTS:
            regions['cr_vbf_qcd_nodijetcut'] = remove_items(regions['cr_vbf_qcd'], ['mjj','detajj','dphijj'])

    # QCD CR to check with deltaphi(jet,MET) cut inverted
    # Will be used to compare the yields with the QCD template obtained from R&S
    if cfg.RUN.REBSMEAR_CHECK:
        regions['cr_vbf_qcd_rs'] = remove_items(regions['sr_vbf'], ['mindphijr'])
        regions['cr_vbf_qcd_rs'] = append_items(regions['cr_vbf_qcd_rs'], ['mindphijr_inv'])
    
    # Dimuon CR
    to_add = ['trig_met', 'at_least_one_tight_mu', 'two_muons', 'dimuon_charge', 'dimuon_mass', 'dpfcalo_cr']
    to_remove = ['veto_muo']
    regions['cr_2m_vbf'] = remove_items( append_items(to_add, common_cuts), to_remove)
    
    # Single muon CR
    to_add = ['trig_met', 'at_least_one_tight_mu', 'one_muon', 'dpfcalo_cr']
    to_remove = ['veto_muo']
    regions['cr_1m_vbf'] = remove_items( append_items(to_add, common_cuts), to_remove)

    # Dielectron CR
    to_add = ['trig_ele', 'at_least_one_tight_el', 'two_electrons', 'dielectron_charge', 'dielectron_mass', 'dpfcalo_cr']
    to_remove = ['veto_ele']
    regions['cr_2e_vbf'] = remove_items( append_items(to_add, common_cuts), to_remove)
    
    # Single electron CR
    to_add = ['trig_ele', 'at_least_one_tight_el', 'one_electron', 'met_el', 'no_el_in_hem', 'dpfcalo_cr']
    to_remove = ['veto_ele']
    regions['cr_1e_vbf'] = remove_items( append_items(to_add, common_cuts), to_remove)

    # Photon CR
    to_add = ['trig_photon', 'at_least_one_tight_photon', 'one_photon', 'photon_pt', 'dpfcalo_cr']
    to_remove = ['veto_photon']
    regions['cr_g_vbf'] = remove_items( append_items(to_add, common_cuts), to_remove)

    # Z CRs with CaloMETNoLep cut
    if cfg.RUN.CALOMET_CHECK:
        for r in ['cr_2e_vbf', 'cr_2m_vbf']:
            regions[f'{r}_calocut'] = append_items(regions[r], 'calo_metptnolep')

    # VBF signal region where the hard-lepton vetoes are replace
    # with lepton veto weights
    to_add = ['met_sr', 'mindphijm']
    to_remove = ['veto_muo', 'veto_tau', 'veto_ele', 'mindphijr', 'recoil']
    regions.update(dict([(f"{region}_no_veto_all", append_items(remove_items(regions[region], to_remove),to_add)) for region in regions.keys() if region.startswith("sr_")]))

    # Region with high detajj cut
    if "sr_vbf_detajj_gt_3p0" in cfg.RUN.EXTRA_REGIONS:
        regions['sr_vbf_detajj_gt_3p0'] = append_items(regions['sr_vbf_no_veto_all'], ['detajj_gt_3p0'])

    # VBF signal region without the dphijj cut
    if "sr_vbf_no_dphijj_cut" in cfg.RUN.EXTRA_REGIONS:
        regions['sr_vbf_no_dphijj_cut'] = remove_items(regions['sr_vbf_no_veto_all'], ['dphijj'])

    if cfg.RUN.TRIGGER_STUDY:
        # Trigger studies
        # num = numerator, den = denominator
        # Single Mu region: Remove mjj cut, add SingleMu trigger, toggle MET trigger
        for cut in ['two_central_jets', 'one_jet_forward_one_jet_central', 'two_hf_jets']:
            regions[f"tr_1m_num_{cut}"] = append_items(remove_items(regions['cr_1m_vbf'], ['recoil']), ['trig_mu', 'mu_pt_trig_safe', cut])
            regions[f"tr_1m_den_{cut}"] = remove_items(regions[f"tr_1m_num_{cut}"], ['trig_met'])

            regions[f"tr_2m_num_{cut}"] = append_items(remove_items(regions['cr_2m_vbf'], ['mjj']), ['trig_mu', 'mu_pt_trig_safe', cut])
            regions[f"tr_2m_den_{cut}"] = remove_items(regions[f"tr_2m_num_{cut}"], ['trig_met'])

        regions[f"tr_g_notrig_num"] = remove_items(regions['cr_g_vbf'], ['recoil', 'photon_pt'])
        regions[f"tr_g_notrig_den"] = remove_items(regions[f"tr_g_notrig_num"], ['trig_photon'])

        for trgname in cfg.TRIGGERS.HT.GAMMAEFF:
            regions[f'tr_g_{trgname}_num'] = append_items(regions[f"tr_g_notrig_num"], [trgname])
            regions[f'tr_g_{trgname}_den'] = append_items(regions[f"tr_g_notrig_den"], [trgname])

            regions[f'tr_g_{trgname}_photon_pt_trig_cut_num'] = append_items(regions[f"tr_g_notrig_num"], [trgname, 'photon_pt_trig'])
            regions[f'tr_g_{trgname}_photon_pt_trig_cut_den'] = append_items(regions[f"tr_g_notrig_den"], [trgname, 'photon_pt_trig'])

    return regions

def ak4_em_frac_weights(weights, diak4, evaluator):
    '''Apply SF for EM fraction cut on jets. Event weight = (Leading jet weight) * (Trailing jet weight)'''
    # Separate weights for the two leading jets:
    # Calculate each jet weight based on the jet pt, if the jet is not in the endcap, assign a weight of 1.
    ak40_in_pos_endcap = ((diak4.i0.eta > 2.5) & (diak4.i0.eta < 3.0)).any()
    ak40_in_neg_endcap = ((diak4.i0.eta > -3.0) & (diak4.i0.eta < -2.5)).any()
    
    ak41_in_pos_endcap = ((diak4.i1.eta > 2.5) & (diak4.i1.eta < 3.0)).any()
    ak41_in_neg_endcap = ((diak4.i1.eta > -3.0) & (diak4.i1.eta < -2.5)).any()

    w_ak40 = np.where(
        ak40_in_pos_endcap,
        evaluator['ak4_em_frac_sf_pos_endcap'](diak4.i0.pt).prod(),
        np.where(
            ak40_in_neg_endcap,
            evaluator['ak4_em_frac_sf_neg_endcap'](diak4.i0.pt).prod(),
            1.
        )
    )

    w_ak41 = np.where(
        ak41_in_pos_endcap,
        evaluator['ak4_em_frac_sf_pos_endcap'](diak4.i1.pt).prod(),
        np.where(
            ak41_in_neg_endcap,
            evaluator['ak4_em_frac_sf_neg_endcap'](diak4.i1.pt).prod(),
            1.
        )
    )

    em_frac_weight = w_ak40 * w_ak41

    weights.add('em_frac_weight', em_frac_weight)

    return weights

def met_trigger_sf(weights, diak4, df, apply_categorized=True):
    '''
    Data/MC SF for the MET trigger, determined as the ratio of 
    two sigmoid functions which are fit to data and MC efficiencies.
    If apply_categorized is set to True, two categories of SF will be applied,
    depending on the leading two jets. Otherwise, one single SF will be applied.
    '''
    year = extract_year(df['dataset'])
    x = df['recoil_pt']
    
    data_params = {
        'two_central_jets' : {
            2017 : (0.048, 165.562, 0.989),
            2018 : (0.046, 174.326, 0.990)
        },
        'mixed' : {
            2017 : (0.044, 165.823, 0.989),
            2018 : (0.041, 179.190, 0.991)
        },
        'inclusive' : {
            2017 : (0.047, 165.573, 0.989),
            2018 : (0.045, 175.975, 0.991)
        }
    }

    mc_params = {
        'two_central_jets' : {
            2017 : (0.050, 150.112, 0.992),
            2018 : (0.052, 157.964, 0.992)
        },
        'mixed' : {
            2017 : (0.045, 150.706, 0.987),
            2018 : (0.045, 160.679, 0.992)
        },
        'inclusive' : {
            2017 : (0.048, 150.315, 0.992),
            2018 : (0.049, 158.786, 0.992)
        }
    }

    if year == 2016:
        sf = np.ones(df.size)
    else:
        if apply_categorized:
            # Two categories: Two central jets & others
            two_central_jets = (diak4.i0.abseta < 2.5) & (diak4.i1.abseta < 2.5)
            two_hf_jets = (diak4.i0.abseta > 3.0) & (diak4.i1.eta > 3.0)
            one_jet_forward_one_jet_central = (~two_central_jets) & (~two_hf_jets)
            
            sf = np.where(
                two_central_jets,
                sigmoid3(x, *data_params['two_central_jets'][year]) / sigmoid3(x, *mc_params['two_central_jets'][year]),
                sigmoid3(x, *data_params['mixed'][year]) / sigmoid3(x, *mc_params['mixed'][year])
            ) 
    
        else:
            sf = sigmoid3(x, *data_params['inclusive'][year]) / sigmoid3(x, *mc_params['inclusive'][year])

    sf[np.isnan(sf) | np.isinf(sf)] == 1
    weights.add("trigger_met", sf)

def apply_hfmask_weights(ak4, weights, evaluator, met_phi, cfg):
    '''On ReReco MC, apply the HF mask efficiency weights.'''
    hfak4 = ak4[(ak4.pt > cfg.RUN.HF_PT_THRESH) & (ak4.abseta > 2.99) & (ak4.abseta < 5.0)]

    # Only consider jets that are back to back with MET
    dphi_hfjet_met = dphi(hfak4.phi, met_phi)
    hfak4 = hfak4[dphi_hfjet_met > 2.5]

    hfweight = evaluator['hf_mask_efficiency_mc'](hfak4.abseta, hfak4.pt).prod()

    weights.add('hfweight', hfweight)

    return weights

def apply_hf_weights_for_qcd_estimation(ak4, weights, evaluator, df, cfg, region):
    '''HF weights for the QCD estimation.'''
    # Don't do anything except for the two QCD regions
    if region not in ['cr_vbf_qcd']:
        return
    hfak4 = ak4[(ak4.pt > cfg.RUN.HF_PT_THRESH) & (ak4.abseta > 2.99) & (ak4.abseta < 5.0)]

    transferfactor = evaluator['hf_mask_noise_passing_rate'](hfak4.abseta, hfak4.pt).prod() / (1 - evaluator['hf_mask_noise_passing_rate'](hfak4.abseta, hfak4.pt).prod())
    transferfactor[np.isnan(transferfactor) | np.isinf(transferfactor)] = 0.

    # CR -> SR transfer factor
    weights.add('hf_qcd_est_weight', transferfactor)

def hfmask_sf(ak4, weights, evaluator, df, cfg):
    '''Apply data/MC SF to account for the HF shape cuts.'''
    hfak4 = ak4[(ak4.pt > cfg.RUN.HF_PT_THRESH) & (ak4.abseta > 2.99) & (ak4.abseta < 5.0)]

    dphi_hfjet_met = dphi(hfak4.phi, df['recoil_phi'])
    dphimask = dphi_hfjet_met > 2.5
    hfak4 = hfak4[dphimask]

    # Truncate the pt since we have limited stats in HF scale factors
    jetmask_1 = (hfak4.abseta > 3.25) & (hfak4.abseta < 4.0)
    jetmask_2 = hfak4.abseta > 4.0
    hfak4.pt[jetmask_1] = np.minimum(299., hfak4.pt[jetmask_1])
    hfak4.pt[jetmask_2] = np.minimum(199., hfak4.pt[jetmask_2])

    sf = evaluator['hf_cuts_sf'](hfak4.abseta, hfak4.pt).prod()
    weights.add('hfmask_sf', sf)
    return weights

def apply_endcap_weights(diak4, weights, evaluator):
    '''Jet eta based weights derived from UL data / ReReco MC.'''
    w_ak40 = evaluator['endcap_w_rereco_mc_leadak4'](diak4.i0.eta).prod()
    w_ak41 = evaluator['endcap_w_rereco_mc_trailak4'](diak4.i1.eta).prod()

    w = w_ak40 * w_ak41

    weights.add('endcap_weight', w)

    return weights

def met_xy_correction(df, met_pt, met_phi):
    '''Apply MET XY corrections (UL based).'''
    import yaml

    correction_src_file = bucoffea_path('data/met/metxycorr.yaml')

    with open(correction_src_file) as f:
        xycorr = yaml.load(f.read(),Loader=yaml.SafeLoader)

    npv = df['PV_npvs']

    met_px = met_pt * np.cos(met_phi)
    met_py = met_pt * np.sin(met_phi)

    def correction(a,b):
        return -(a * npv + b)

    dataset = df['dataset']
    year = extract_year(df['dataset'])

    # Get the correction factors, depending on the run (if data)
    if df['is_data']:
        # Extract the run information from the dataset name
        run = re.findall('201\d([A-F])', dataset)[0]
        # Get the corrections for this run
        xycorrections = xycorr[year][run]

    else:
        # No run info needed, just extract the corrections for MC, based on year
        xycorrections = xycorr[year]['MC']

    # Extract the coefficients for the X and Y corrections
    xcorr_coef = ( xycorrections['X']['a'], xycorrections['X']['b'] )
    ycorr_coef = ( xycorrections['Y']['a'], xycorrections['Y']['b'] )

    met_xcorr = correction(*xcorr_coef)
    met_ycorr = correction(*ycorr_coef)

    corr_met_px = met_px + met_xcorr
    corr_met_py = met_py + met_ycorr

    corr_met_pt = np.hypot(corr_met_px, corr_met_py)
    corr_met_phi = np.arctan2(corr_met_py, corr_met_px)

    return corr_met_pt, corr_met_phi

def pileup_sf_variations(df, evaluator, cfg):
    if cfg.SF.PILEUP.MODE == 'nano':
        pu_weights = {
            "puSFUp" : df['puWeightUp'],
            "puSFDown" : df['puWeightDown'],
            "puSFNom" : df['puWeight'],
        }
    else:
        raise NotImplementedError(f'No implementation for cfg.PILEUP.MODE: {cfg.SF.PILEUP.MODE}')

    return pu_weights
