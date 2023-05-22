def get_files_per_job(dataset,time_per_job = 3):
    # nfile per hour of running time
    filesperjob_dict ={ 'GluGlu_HToInvisible': 1,
                        'ttH_HToInvisible': 1,
                        'VBF_HToInvisible': 1,
                        'WminusH_WToQQ_HToInvisible': 1,
                        'WplusH_WToQQ_HToInvisible': 1,
                        'ZH_ZToQQ_HToInvisible': 1,
                        'DYJetsToLL_LHEFilterPtZ-0To50': 10,
                        'DYJetsToLL_LHEFilterPtZ-50To100': 7,
                        'DYJetsToLL_LHEFilterPtZ-100To250': 5,
                        'DYJetsToLL_LHEFilterPtZ-250To400': 1,
                        'DYJetsToLL_LHEFilterPtZ-400To650': 1,
                        'DYJetsToLL_LHEFilterPtZ-650ToInf': 1,
                        'EWKWMinus2Jets_WToLNu': 4,
                        'EWKWPlus2Jets_WToLNu': 3,
                        'EWKZ2Jets_ZToLL': 3,
                        'EWKZ2Jets_ZToNuNu': 4,
                        'GJets_DR-0p4': 30,
                        'G1Jet': 30,
                        'VBFGamma_5f_DipoleRecoil-mg': 10,
                        'WJetsToLNu_Pt-100To250': 3,
                        'WJetsToLNu_Pt-250To400': 1,
                        'WJetsToLNu_Pt-400To600': 1,
                        'WJetsToLNu_Pt-600ToInf': 1,
                        'Z1JetsToNuNu': 1,
                        'Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150': 15,
                        'Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250': 1,
                        'Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400': 1,
                        'Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf': 1,
                        'EGamma': 8,
                        'MET': 3,
                        }
    # Look up files per job
    keys_contained = [key for key in filesperjob_dict.keys() if key in dataset]
    filesperjob = -1
    if len(keys_contained)==1:
        filesperjob = int(round(filesperjob_dict[keys_contained[0]] * time_per_job))
        if 'Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400' in dataset:
            filesperjob = 1
    return filesperjob