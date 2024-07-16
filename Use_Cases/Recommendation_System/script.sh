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

#curl -O https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip

python3 process_rawfile.py --dataset epinions --directory Users/shrabanighosh/My\ work/UNCC/Summer\ 2024/itsc-2214-readings-main/Community-Detection-in-Large-Graphs-and-Applications/Use_Cases/Recommendatin_System/

