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

curl -O http://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip

unzip ciao.zip

python3 preprocess.py --dataset ciao

#python3 ../jsontocsv.py --dataset ciao --inputfilename ego_splitting_res2.5_min10.json
#
python3 subgraph_create.py --dataset ciao --graphfile renumbered_graph_epinions.csv --inputfilename louvain_ciao.csv --overlapping nonoverlapping --outdirectory louvain
#
#python3 create_node_propensity.py --inDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/louvain --outDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/propensity_subgraph_louvain
#
#python3 find_center_of_communities.py --directory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/propensity_subgraph_louvain
#
#python3 trust_prediction.py --graphfile renumbered_graph_ciao.csv --communityfile community_clusters/spectral_ciao_c10.csv --community_center propensity_subgraph_spectral/centerclusters.csv --ratingfile ciao_rating.csv --overlap nonoverlap
#
