#!/usr/bin/env python

from bucoffea.helpers.dataset import extract_year
from bucoffea.processor.executor import run_uproot_job_nanoaod
from bucoffea.helpers.cutflow import print_cutflow
from coffea.util import save
import coffea.processor as processor
import argparse

def parse_commandline():

    parser = argparse.ArgumentParser()
    parser.add_argument('processor', type=str, help='The processor to be run. (monojet or vbfhinv)')
    args = parser.parse_args()

    return args

def main():
    fileset = {
        "EGamma_ver2_2018D" : [
            "/eos/cms/store/group/phys_higgs/vbfhiggs/PFNANO_V9_17Feb23_PostNanoTools/EGamma/EGamma_ver2_2018D/230306_160844/0001/tree_1648.root"
        ],
        "GluGlu_HToInvisible_M125_HiggspTgt190_pow_pythia8_2018" : [
            "/eos/cms/store/group/phys_higgs/vbfhiggs/PFNANO_V9_17Feb23_PostNanoTools/GluGlu_HToInvisible_M125_HiggspTgt190_TuneCP5_13TeV_powheg_pythia8/GluGlu_HToInvisible_M125_HiggspTgt190_pow_pythia8_2018/230303_181428/0000/tree_10.root"
        ],
        "Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20-amcatnloFXFX_2018" : [
            "/eos/cms/store/group/phys_higgs/vbfhiggs/PFNANO_V9_17Feb23_PostNanoTools/Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20-amcatnloFXFX_2018/230303_184057/0000/tree_11.root"
        ]
    }

    years = list(set(map(extract_year, fileset.keys())))
    assert(len(years)==1)

    args = parse_commandline()
    processor_class = args.processor

    if processor_class == 'monojet':
        from bucoffea.monojet import monojetProcessor
        processorInstance = monojetProcessor()
    elif processor_class == 'vbfhinv':
        from bucoffea.vbfhinv import vbfhinvProcessor
        processorInstance = vbfhinvProcessor()
    elif processor_class == 'lhe':
        from bucoffea.gen.lheVProcessor import lheVProcessor
        processorInstance = lheVProcessor()
    elif args.processor == 'purity':
        from bucoffea.photon_purity import photonPurityProcessor
        processorInstance = photonPurityProcessor()
    elif args.processor == 'sumw':
        from bucoffea.gen import mcSumwProcessor
        processorInstance = mcSumwProcessor()
    elif args.processor == 'gen':
        from bucoffea.gen.genVbfProcessor import genVbfProcessor
        processorInstance = genVbfProcessor()

    for dataset, filelist in fileset.items():
        newlist = []
        for file in filelist:
            if file.startswith("/store/"):
                newlist.append("root://cms-xrd-global.cern.ch//" + file)
            else: newlist.append(file)
        fileset[dataset] = newlist

    for dataset, filelist in fileset.items():
        tmp = {dataset:filelist}
        output = run_uproot_job_nanoaod(tmp,
                                    treename='Runs' if args.processor=='sumw' else 'Events',
                                    processor_instance=processorInstance,
                                    executor=processor.futures_executor,
                                    executor_args={'workers': 1, 'flatten': True},
                                    chunksize=50000,
                                    )
        save(output, f"{processor_class}_{dataset}.coffea")
        # Debugging / testing output
        # debug_plot_output(output)
        print_cutflow(output, outfile=f'{processor_class}_cutflow_{dataset}.txt')

if __name__ == "__main__":
    main()
