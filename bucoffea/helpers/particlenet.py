import numpy as np
import pandas as pd
import awkward as ak
import onnxruntime as ort

from coffea.analysis_objects import JaggedCandidateArray
from bucoffea.helpers import object_overlap



def load_particlenet_model(path):
    """
    Given the path to the .onnx file which has the model,
    loads the ParticleNet model into runtime and returns it.
    """
    return ort.InferenceSession(path)


def run_particlenet_model(session, inputs):
    return session.run(None, inputs)[0]


def load_pf_cands(df, objects_to_clean, dr_min=0.1):
    pfcands = JaggedCandidateArray.candidatesfromcounts(
        df['nPFCand'],
        pt=df['PFCand_pt'],
        eta=df['PFCand_eta'],
        phi=df['PFCand_phi'],
        mass=df['PFCand_mass'],
        energy=df['PFCand_energy'],
        pdgId=df['PFCand_pdgId'],
        charge=df['PFCand_charge'],
        puppiw=df['PFCand_puppiWeight'],
    )
    for objs in objects_to_clean:
        pfcands = pfcands[object_overlap(pfcands, objs, dr=dr_min)]
    return pfcands


def build_particlenet_inputs(pfcands, mask, num_cands=100, sortby='pt'):
    """
    Function that returns the dictionary with the PF candidate features,
    as to be read by the ParticleNet NN.
    """
    masked_pfcands = pfcands[mask]

    # PF candidates are sorted by pt in custom NanoAOD
    # If we're going to sort by something else, do the sorting here
    if sortby != 'pt':
        indices = getattr(masked_pfcands, sortby).argsort()
        masked_pfcands = masked_pfcands[indices]

    def get_feature_array(feature):
        """
        Helper function to get a 2D array for a single feature.
        """
        if feature == 'energy_log':
            arr = np.log(masked_pfcands.energy)
        elif feature == 'pt_log':
            arr = np.log(masked_pfcands.pt)
        else:
            arr = getattr(masked_pfcands, feature)
        
        # If the length of the array is less than num_cands,
        # need to pad the rest of the array with zeros
        def get_first_num_cands(x):
            if len(x) >= num_cands:
                return x[:num_cands]
            
            # Pad the array with zeros if necessary
            return np.pad(x[:num_cands], (0, num_cands-len(x)), 'constant', constant_values=(0, 0))

        return np.array(list(map(get_first_num_cands, arr)))

    # Retrieve the features
    feature_list = ['pt', 'eta', 'phi', 'energy', 
        'pdgId', 'charge', 'puppiw', 'energy_log', 'pt_log']

    features = {}
    for feature in feature_list:
        features[feature] = get_feature_array(feature) 

    # Stack all the features together
    pf_features = np.array(tuple(features[x] for x in feature_list))
    pf_points = np.array((features['eta'], features['phi']))
    pf_mask = np.float32([features['pt'] != 0])

    # Swap axes 0 and 1 to obtain correct shape
    pf_features = np.moveaxis(pf_features, 0, 1)
    pf_points = np.moveaxis(pf_points, 0, 1)
    pf_mask = np.moveaxis(pf_mask, 0, 1)

    # Construct the inputs required by the ParticleNet
    inputs = {}
    inputs["pf_features"] = pf_features.astype(np.float32)
    inputs["pf_points"] = pf_points.astype(np.float32)
    inputs["pf_mask"] = pf_mask

    return inputs