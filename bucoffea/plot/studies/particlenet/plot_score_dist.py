#!/usr/bin/env python

import os
import sys
import re

from matplotlib import pyplot as plt

from coffea.util import load
from coffea import hist


def plot_score_dist(acc, distribution, region, dataset):
    """
    Helper function to plot ParticleNet score distribution.
    """
    h = acc[distribution]
    
    # Rebinning
    new_ax = hist.Bin("score", "ParticleNet VBF-like Score", 25, 0, 1)
    h = h.rebin('score', new_ax)

    # Integrate region and score type
    h = h.integrate('region', region).integrate('score_type', 'VBF-like')

    # Plot!
    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='dataset')

    fig.savefig(f"{distribution}.pdf")


def main():
    inpath = sys.argv[1]
    acc = load(inpath)

    plot_score_dist(acc,
        distribution="particlenet_score",
        region="inclusive",
        dataset="VBF_HToInvisible.*M125.*2017",
    )

if __name__ == '__main__':
    main()