import os, glob
import numpy as np

submitted = []
info = {}
nsub = 0
samples = ['GluGlu_HToInvisible',
            'ttH_HToInvisible',
            'VBF_HToInvisible',
            'WminusH_WToQQ_HToInvisible',
            'WplusH_WToQQ_HToInvisible',
            'ZH_ZToQQ_HToInvisible',
            'DYJetsToLL_LHEFilterPtZ-0To50',
            'DYJetsToLL_LHEFilterPtZ-50To100',
            'DYJetsToLL_LHEFilterPtZ-100To250',
            'DYJetsToLL_LHEFilterPtZ-250To400',
            'DYJetsToLL_LHEFilterPtZ-400To650',
            'DYJetsToLL_LHEFilterPtZ-650ToInf',
            'EWKWMinus2Jets_WToLNu',
            'EWKWPlus2Jets_WToLNu',
            'EWKZ2Jets_ZToLL',
            'EWKZ2Jets_ZToNuNu',
            'GJets_DR-0p4',
            'VBFGamma_5f_DipoleRecoil-mg',
            'WJetsToLNu_Pt-100To250',
            'WJetsToLNu_Pt-250To400',
            'WJetsToLNu_Pt-400To600',
            'WJetsToLNu_Pt-600ToInf',
            'Z1JetsToNuNu',
            'Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150',
            'Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250',
            'Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400',
            'Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf',
            'EGamma',
            'MET',
]
for sample in samples:
    info[sample] = {}
    temp = []
    times = []
    for fname in glob.glob("submission/PFNANO_V9_17Feb23_PostNanoTools/files/input_*"+sample+"*txt"):
        with open(fname) as f_:
            files = [x.replace('\n','') for x in f_.readlines()]
            nfiles = len(files)
            temp += files
        if not os.path.exists(fname.replace('input_','err_')): continue
        with open(fname.replace('input_','err_')) as f_:
            t = [x.replace('\n','') for x in f_.readlines() if 'user' in x]
            if len(t)==0: continue
            t = t[0].replace('user\t','')
            times.append(float(t[:t.find('m')])/nfiles)
    info[sample]['nfiles'] = len(temp)
    info[sample]['nt'] = len(times)
    if len(temp)==0:
        print(sample, temp,times, glob.glob("submission/PFNANO_V9_17Feb23_PostNanoTools/files/input_*"+sample+"*txt"))
        continue
    info[sample]['times'] = np.average(times)
    info[sample]['err'] = np.std(times)
    info[sample]['min'] = np.min(times)
    info[sample]['max'] = np.max(times)
    info[sample]['exp'] = np.max(times)*info[sample]['nfiles']
    info[sample]['split'] = max(1, info[sample]['exp']/(60),0)
    info[sample]['filesperjob'] = max(1, 1.*info[sample]['nfiles']/info[sample]['split'])
    nsub += info[sample]['split']
    submitted += temp 

all_files = glob.glob("/eos/cms/store/group/phys_higgs/vbfhiggs/PFNANO_V9_17Feb23_PostNanoTools/*/*/*/*/*root")
print(f"Non analysed {len(list(set(all_files)-set(submitted)))} out of {len(all_files)} files")

hours = 1
nsub_tot = 0
for sample in info.keys():
    nsub = int(round(max(1,info[sample]['split']/hours),0))
    nsub_tot += nsub
    n_files = int(round(info[sample]['filesperjob']*hours,0))
    print(f"Split by {n_files} file can be used for sample:{sample} to produce {nsub} jobs")
print("nsub", nsub_tot)

