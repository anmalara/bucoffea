#!/usr/bin/env python

import os
import sys
import re

from matplotlib import pyplot as plt

from coffea.util import load
from coffea import hist
from klepto.archives import dir_archive

from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi

pjoin = os.path.join


DATASET_LABELS = {
    "VBF_HToInvisible.*M125.*" : "VBFH(inv)",
    "GluGlu.*HToInvisible.*M125.*" : "ggH(inv)",
}


def plot_score_dist(acc, distribution, region, outdir):
    """
    Helper function to plot ParticleNet score distribution.
    """
    acc.load(distribution)
    h = acc[distribution]
    
    # Merge datasets, scale by xs * lumi / sumw
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebinning
    new_ax = hist.Bin("score", "ParticleNet VBF-like Score", 25, 0, 1)
    h = h.rebin('score', new_ax)

    # Integrate region and score type
    h = h.integrate('region', region).integrate('score_type', 'VBF-like')

    # Plot!
    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='dataset')

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        for dregex, dlabel in DATASET_LABELS.items():
            if re.match(dregex, label):
                handle.set_label(dlabel)
                break

    ax.legend(handles=handles, title="Dataset (2018)")

    outpath = pjoin(outdir, f"{distribution}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load("sumw")

    outtag = inpath.rstrip('/').split('/')[-1]

    outdir = f"./output/{outtag}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plot_score_dist(acc,
        distribution="particlenet_score",
        region="sr_vbf_no_veto_all",
        outdir=outdir,
    )

if __name__ == '__main__':
    main()