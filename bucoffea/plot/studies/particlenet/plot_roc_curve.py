#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

pjoin = os.path.join


def plot_roc_curve(f, outdir, treename='sr_vbf_no_veto_all'):
    """Plot ROC curve for the given ROOT file and the tree."""
    tree = f[treename]

    scores = tree["particleNet_vbfScore"].array()
    true_labels = tree["particleNet_label"].array()
    mjj = tree["mjj"].array()

    weights = tree["weight_total"].array()

    # Compute ROC for ParticleNet score and mjj
    mjj_norm = mjj / mjj.max()
    fpr_pnet, tpr_pnet, th_pnet = roc_curve(true_labels, scores, sample_weight=weights)
    fpr_mjj,  tpr_mjj,  th_mjj  = roc_curve(true_labels, mjj_norm, sample_weight=weights)

    # Sort FPR and TPR, so that AUC calculation below works fine
    pnet_sorted = np.argsort(fpr_pnet)
    mjj_sorted = np.argsort(fpr_mjj)

    auc_pnet = auc(fpr_pnet[pnet_sorted], tpr_pnet[pnet_sorted])
    auc_mjj = auc(fpr_mjj[mjj_sorted], tpr_mjj[mjj_sorted])

    # Plot the ROC curves for mjj and ParticleNet
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr_pnet, 
        tpr=tpr_pnet, 
        estimator_name="ParticleNet", 
        roc_auc=auc_pnet).plot(ax=ax)
    
    RocCurveDisplay(fpr=fpr_mjj,  
        tpr=tpr_mjj,  
        estimator_name=r"$m_{jj}$ Cut", 
        roc_auc=auc_mjj).plot(ax=ax)

    ax.legend(title="Classifier")

    ax.text(0,1,"VBF H(inv) SR",
        fontsize=14,
        ha="left",
        va="bottom",
        transform=ax.transAxes
    )

    ax.text(1,1,"Run2 2018",
        fontsize=14,
        ha="right",
        va="bottom",
        transform=ax.transAxes
    )

    outpath = pjoin(outdir, "roc.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    # Path to the merged ROOT file
    inpath = sys.argv[1]

    outtag = inpath.rstrip('/').split('/')[-2]

    outdir = f"./output/{outtag}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    f = uproot.open(inpath)

    plot_roc_curve(f, outdir=outdir)

if __name__ == '__main__':
    main()