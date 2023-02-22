#!/usr/bin/env python

import os
import sys
import re

from matplotlib import pyplot as plt

from coffea.util import load
from coffea import hist
from klepto.archives import dir_archive
from tqdm import tqdm

from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi

pjoin = os.path.join


DATASET_LABELS = {
    "VBF_HToInvisible.*M125.*" : "VBF H(inv)",
    "GluGlu.*HToInvisible.*M125.*" : "ggH(inv)",
    "EWKZ2Jets.*ZToNuNu.*" : "EWK Z(vv)",
}


def plot_score_dist(acc, distribution, region, outdir, dataset):
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
    hist.plot1d(h[re.compile(dataset)], ax=ax, overlay='dataset')

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        for dregex, dlabel in DATASET_LABELS.items():
            if re.match(dregex, label):
                handle.set_label(dlabel)
                break

    ax.legend(handles=handles, title="Dataset (2018)")

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

    outpath = pjoin(outdir, f"{region}_{distribution}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def compare_ggh_score_dist_with_high_detajj(acc, outdir, distribution, dataset="GluGlu_HToInvisible.*M125.*2018"):
    """
    Compares ggH(inv) score distributions between the VBF SR and
    VBF SR + detajj>3 cut.
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

    h = h.integrate("dataset", re.compile(dataset)).integrate("score_type", "VBF-like")

    regions_labels = {
        "sr_vbf_no_veto_all" : "VBF H(inv) SR",
        "sr_vbf_detajj_gt_3p0" : r"SR + $\Delta\eta_{jj} > 3$",
    }

    fig, ax = plt.subplots()
    legend_labels = []

    for region, label in regions_labels.items():
        hist.plot1d(
            h.integrate("region", region),
            ax=ax,
            clear=False
        )
        legend_labels.append(label)
    
    ax.legend(title="Region", labels=legend_labels)

    ax.text(0,1,"ggH(inv)",
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

    outpath = pjoin(outdir, "ggH_sr_detajj_cut.pdf")
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

    regions = ["sr_vbf_no_veto_all"]

    # Regular expression matching all the datasets we're interested in
    dataset = "(VBF_HToInvisible|GluGlu_HToInvisible|EWKZ2Jets.*ZToNuNu).*2018"


    for region in tqdm(regions, desc="Plotting score distributions"):
        plot_score_dist(acc,
            distribution="particlenet_score",
            region=region,
            outdir=outdir,
            dataset=dataset,
        )


if __name__ == '__main__':
    main()