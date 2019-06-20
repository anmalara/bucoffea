from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from awkward import JaggedArray
import numpy as np

import lz4.frame as lz4f
import cloudpickle
from copy import deepcopy
import os
pjoin = os.path.join
from collections import defaultdict
os.environ["ENV_FOR_DYNACONF"] = "era2016"
os.environ["SETTINGS_FILE_FOR_DYNACONF"] = os.path.abspath("config.yaml")
from dynaconf import settings as cfg

def setup_candidates(df):
    muons = JaggedCandidateArray.candidatesfromcounts(
        df['nMuon'],
        pt=df['Muon_pt'],
        eta=df['Muon_eta'],
        phi=df['Muon_phi'],
        mass=df['Muon_mass'],
        charge=df['Muon_charge'],
        mediumId=df['Muon_mediumId'],
        iso=df["Muon_pfRelIso04_all"],
        tightId=df['Muon_tightId']
    )

    electrons = JaggedCandidateArray.candidatesfromcounts(
        df['nElectron'],
        pt=df['Electron_pt'],
        eta=df['Electron_eta'],
        phi=df['Electron_phi'],
        mass=df['Electron_mass'],
        charge=df['Electron_charge'],
        looseId=(df['Electron_cutBased']>=1),
        tightId=(df['Electron_cutBased']==4)
    )
    taus = JaggedCandidateArray.candidatesfromcounts(
        df['nTau'],
        pt=df['Tau_pt'],
        eta=df['Tau_eta'],
        phi=df['Tau_phi'],
        mass=df['Tau_mass'],
        decaymode=df['Tau_idDecayModeNewDMs'],
        clean=df['Tau_cleanmask'],
        iso=df['Tau_idMVAnewDM2017v2'],
    )
    jets = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df['Jet_pt'],
        eta=df['Jet_eta'],
        phi=df['Jet_phi'],
        mass=df['Jet_mass'],

        # Jet ID bit mask:
        # Bit 0 = Loose
        # Bit 1 = Tight
        tightId=(df['Jet_jetId']&2) == 2,
        csvv2=df["Jet_btagCSVV2"],
        deepcsv=df['Jet_btagDeepB'],
        # nef=df['Jet_neEmEF'],
        nhf=df['Jet_neHEF'],
        chf=df['Jet_chHEF'],
        clean=df['Jet_cleanmask']
        # cef=df['Jet_chEmEF'],
    )
    return jets, muons, electrons, taus

def define_dphi_jet_met(jets, met_phi, njet=4, ptmin=30):
    """Calculate minimal delta phi between jets and met

    :param jets: Jet candidates to use, must be sorted by pT
    :type jets: JaggedCandidateArray
    :param met_phi: MET phi values, one per event
    :type met_phi: array
    :param njet: Number of leading jets to consider, defaults to 4
    :type njet: int, optional
    """

    # Use the first njet jets with pT > ptmin
    jets=jets[jets.pt>30]
    jets = jets[:,:njet]

    dphi = np.abs((jets.phi - met_phi + np.pi) % (2*np.pi)  - np.pi)

    return dphi.min()

class monojetProcessor(processor.ProcessorABC):
    def __init__(self, year="2018"):
        self.year=year
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        met_axis = hist.Bin("met", r"$p_{T}^{miss}$ (GeV)", 100, 0, 1000)
        jet_pt_axis = hist.Bin("jetpt", r"$p_{T}$ (GeV)", 100, 0, 1000)
        jet_eta_axis = hist.Bin("jeteta", r"$\eta$ (GeV)", 50, -5, 5)
        dpfcalo_axis = hist.Bin("dpfcalo", r"$1-Calo/PF$", 20, -1, 1)
        btag_axis = hist.Bin("btag", r"B tag discriminator", 20, 0, 1)
        multiplicity_axis = hist.Bin("multiplicity", r"multiplicity", 10, -0.5, 9.5)
        dphi_axis = hist.Bin("dphi", r"$\Delta\phi$", 50, 0, 2*np.pi)

        self._accumulator = processor.dict_accumulator({
            "met" : hist.Hist("Counts", dataset_axis, met_axis),
            "jet0pt" : hist.Hist("Counts", dataset_axis, jet_pt_axis),
            "jet0eta" : hist.Hist("Counts", dataset_axis, jet_eta_axis),
            "jetpt" : hist.Hist("Counts", dataset_axis, jet_pt_axis),
            "jeteta" : hist.Hist("Counts", dataset_axis, jet_eta_axis),
            "dpfcalo" : hist.Hist("Counts", dataset_axis, dpfcalo_axis),
            "jetbtag" : hist.Hist("Counts", dataset_axis, btag_axis),
            "jet_mult" : hist.Hist("Jets", dataset_axis, multiplicity_axis),
            "bjet_mult" : hist.Hist("B Jets", dataset_axis, multiplicity_axis),
            "loose_ele_mult" : hist.Hist("Loose electrons", dataset_axis, multiplicity_axis),
            "tight_ele_mult" : hist.Hist("Tight electrons", dataset_axis, multiplicity_axis),
            "loose_muo_mult" : hist.Hist("Loose muons", dataset_axis, multiplicity_axis),
            "tight_muo_mult" : hist.Hist("Tight muons", dataset_axis, multiplicity_axis),
            "veto_tau_mult" : hist.Hist("Veto taus", dataset_axis, multiplicity_axis),
            "dphijm" : hist.Hist("min(4 leading jets, MET)", dataset_axis, dphi_axis),
            "cutflow_sr_j": processor.defaultdict_accumulator(int),
            "cutflow_sr_v": processor.defaultdict_accumulator(int)
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):


        # Lepton candidates
        jets, muons, electrons, taus = setup_candidates(df)
        loose_muons = muons[muons.mediumId & (muons.pt>10) & (muons.iso < 0.25)]
        tight_muons = muons[muons.mediumId & (muons.pt>20) & (muons.iso < 0.15)]
        loose_electrons = electrons[electrons.looseId & (electrons.pt>10)]
        tight_electrons = electrons[electrons.tightId & (electrons.pt>20)]

        # Jets
        clean_jets = jets[jets.clean==1]
        jet_acceptance = (clean_jets.eta<2.4)&(clean_jets.eta>-2.4)
        jet_fractions = (clean_jets.chf>0.1)&(clean_jets.nhf<0.8)


        btag_cut = cfg.BTAG.CUTS[cfg.BTAG.algo][cfg.BTAG.wp]

        jet_btagged = getattr(clean_jets, cfg.BTAG.algo) > btag_cut
        bjets = clean_jets[jet_acceptance & jet_btagged]


        goodjets = clean_jets[jet_fractions \
                              & jet_acceptance \
                              & jet_btagged==0 \
                              & clean_jets.tightId ]

        # Taus
        veto_taus = taus[ (taus.clean==1) \
                         & (taus.decaymode) \
                         & (taus.pt > cfg.TAU.CUTS.PT)\
                         & (np.abs(taus.eta) < cfg.TAU.CUTS.ETA) \
                         & ((taus.iso&2)==2)]

        # MET
        df["dPFCalo"] = 1 - df["CaloMET_pt"] / df["MET_pt"]
        df["minDPhiJetMet"] = define_dphi_jet_met(goodjets, df['MET_phi'], njet=4, ptmin=30)

        # Selection
        # TODO:
        #   Photons
        # Naming syntax:
        # sr = signal region
        # j = monojet
        # v = mono-V
        # -> "sr_j" = Monojet signal region
        selections = defaultdict(processor.PackedSelection)
        selections['inclusive'].add("all", np.ones(df.size)==1)

        selections["sr_j"].add('filt_met', df['Flag_METFilters'])
        selections["sr_j"].add('trig_met', df[cfg.TRIGGERS.MET])
        selections["sr_j"].add('veto_ele', loose_electrons.counts==0)
        selections["sr_j"].add('veto_muo', loose_muons.counts==0)
        selections["sr_j"].add('veto_photon',np.ones(df.size)==1) # TODO
        selections["sr_j"].add('veto_tau',veto_taus.counts==0)
        selections["sr_j"].add('veto_b',bjets.counts==0)
        selections["sr_j"].add('leadjet_pt_eta', (jets.pt[:,0] > cfg.SELECTION.SIGNAL.LEADJET.PT) \
                                                 & (np.abs(jets.eta[:,0]) < cfg.SELECTION.SIGNAL.LEADJET.ETA))
        selections["sr_j"].add('leadjet_id',jets[:,0].tightId)
        selections["sr_j"].add('dphijm',df['minDPhiJetMet'] > cfg.SELECTION.SIGNAL.MINDPHIJM)
        selections["sr_j"].add('dpfcalo',np.abs(df['dPFCalo']) < cfg.SELECTION.SIGNAL.DPFCALO)
        selections["sr_j"].add('met_signal', df['MET_pt']>cfg.SELECTION.SIGNAL.MET)

        selections["sr_v"] = deepcopy(selections["sr_j"])
        selections["sr_v"].add("tau21", np.ones(df.size)==1)

        
        
        output = self.accumulator.identity()

        
        for seltag, selection in selections.items():

            # Cutflow plot for signal and control regions
            if any(x in seltag for x in ["sr", "cr"]):
                output['cutflow_' + seltag]['all']+=df.size
                for icut, cutname in enumerate(selection.names):
                    output['cutflow_' + seltag][cutname] += selection.all(*selection.names[:icut+1]).sum()

            dataset = seltag
            mask = selection.all(*selection.names)

            # Multiplicities
            def fill_mult(name, candidates):
                output[name].fill(dataset=dataset, multiplicity=candidates[mask].counts)

            fill_mult('jet_mult', clean_jets)
            fill_mult('bjet_mult',bjets)
            fill_mult('loose_ele_mult',loose_electrons)
            fill_mult('tight_ele_mult',tight_electrons)
            fill_mult('loose_muo_mult',loose_muons)
            fill_mult('tight_muo_mult',tight_muons)
            fill_mult('veto_tau_mult',veto_taus)

            # All jets
            output['jeteta'].fill(dataset=dataset,
                                    jeteta=jets[mask].eta.flatten())
            output['jetpt'].fill(dataset=dataset,
                                    jetpt=jets[mask].pt.flatten())

            # Leading jet (has to be in acceptance)
            output['jet0eta'].fill(dataset=dataset,
                                    jeteta=jets[mask].eta[:,0].flatten())
            output['jet0pt'].fill(dataset=dataset,
                                    jetpt=jets[mask].pt[:,0].flatten())

            # B tag discriminator
            output['jetbtag'].fill(dataset=dataset,
                                    btag=getattr(clean_jets[mask&jet_acceptance], cfg.BTAG.algo).flatten())

            # MET
            output['dpfcalo'].fill(dataset=dataset,
                                    dpfcalo=df["dPFCalo"][mask])
            output['met'].fill(dataset=dataset,
                                    met=df["MET_pt"][mask])
            output['dphijm'].fill(dataset=dataset,
                                    dphi=df["minDPhiJetMet"][mask])
        return output

    def postprocess(self, accumulator):
        return accumulator


def main():
    fileset = {
        # "Znunu_ht200to400" : [
        #     "./data/FFD69E5A-A941-2D41-A629-9D62D9E8BE9A.root"
        # ],
        # "NonthDM" : [
        #     "./data/24EE25F5-FB54-E911-AB96-40F2E9C6B000.root"
        # ]
        "TTbarDM" : [
            "./data/A13AF968-8A88-754A-BE73-7264241D71D5.root"
        ]
    }

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist

    output = processor.run_uproot_job(fileset,
                                  treename='Events',
                                  processor_instance=monojetProcessor(2017),
                                  executor=processor.futures_executor,
                                  executor_args={'workers': 4, 'function_args': {'flatten': True}},
                                  chunksize=500000,
                                 )
    with lz4f.open("hists.cpkl.lz4", mode="wb", compression_level=5) as fout:
        cloudpickle.dump(output, fout)

    # Debugging / testing output
    debug_plot_output(output)
    debug_print_cutflows(output)

def debug_plot_output(output):
    """Dump all histograms as PDF."""
    outdir = "out"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for name in output.keys():
        if name.startswith("_"):
            continue
        if name.startswith("cutflow"):
            continue
        fig, ax, _ = hist.plot1d(output[name],overlay="dataset",overflow='all')
        fig.suptitle(name)
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.1, 1e8)
        fig.savefig(pjoin(outdir, "{}.pdf".format(name)))



def debug_print_cutflows(output):
    """Pretty-print cutflow data to the terminal."""
    import tabulate
    for cutflow_name in [ x for x in output.keys() if x.startswith("cutflow")]:
        table = []
        print("----")
        print(cutflow_name)
        print("----")
        for cut, count in sorted(output[cutflow_name].items(), key=lambda x:x[1], reverse=True):
            table.append([cut, count])
        print(tabulate.tabulate(table, headers=["Cut", "Passing events"]))



if __name__ == "__main__":
    main()