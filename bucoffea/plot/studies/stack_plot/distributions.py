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