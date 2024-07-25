#!/bin/bash
#urls=(
#    "https://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip"
#    "https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"
#    # Add more URLs here
#)
#
#for url in "${urls[@]}"; do
#    curl -O "$url"
#done

curl -O https://snap.stanford.edu/data/email-Eu-core.txt.gz

gunzip email-Eu-core.txt.gz

python3 email_network.py --graphfile email-Eu-core.txt --inputdir community_clusters --outdir nonoverlapping_metrics --overlapping False

python3 email_network.py --graphfile email-Eu-core.txt --inputdir overlapping --overlapping overlapping --outdir overlapping_metrics

python3 plot_metrics.py --inputdir overlapping_metrics

python3 plot_metrics.py --inputdir nonoverlapping_metrics

python3 overlapping_density.py --filename overlapping/ego-splitting-email-Eu-core.csv

