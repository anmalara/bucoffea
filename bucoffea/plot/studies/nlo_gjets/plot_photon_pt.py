#!/usr/bin/env python

import os
import sys
import re

from matplotlib import pyplot as plt
from coffea import hist

from bucoffea.plot.util import merge_extensions, scale_xs_lumi, merge_datasets
from klepto.archives import dir_archive

pjoin = os.path.join


def plot_photon_pt(acc, outdir, distribution="photon_pt0", dataset="G1Jet.*LHEGpT.*", region="cr_g_vbf"):
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)

    if distribution == "genvpt_check":
        h = h[re.compile(dataset)].integrate("type","Nano")
    else:
        h = h.integrate("region", region)[re.compile(dataset)]

    fig, ax = plt.subplots()

    hist.plot1d(h, ax=ax, overlay="dataset", stack=True)

    ax.set_yscale("log")
    ax.set_ylim(1e0,1e6)

    if distribution == "genvpt_check":
        ax.set_xlabel(r"GEN Photon $p_T$ (GeV)")
    else:
        ax.set_xlim(left=230)
        ax.set_xlabel(r"Offline Photon $p_T$ (GeV)")

    ax.text(0,1,r"$\gamma$ + jets CR",
        fontsize=14,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    
    ax.text(1,1,"2018",
        fontsize=14,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    outpath = pjoin(outdir, f"nlo_gjets_{distribution}.pdf")
    fig.savefig(outpath)
    plt.close(fig)


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load("sumw")

    outdir = pjoin('./output/',list(filter(lambda x:x,inpath.split('/')))[-1])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for distribution in ["photon_pt0", "genvpt_check"]:
        plot_photon_pt(acc, outdir, distribution=distribution)

if __name__ == "__main__":
    main()