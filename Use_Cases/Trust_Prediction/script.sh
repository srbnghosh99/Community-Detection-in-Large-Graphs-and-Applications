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

python3 ../jsontocsv.py --dataset ciao --inputfilename community_clusters/ego_splitting_res2.5_min10.json

python3 subgraph_create.py --dataset ciao --graphfile renumbered_graph_ciao.csv --cdfile louvain_ciao.csv --outdirectory louvain --overlap nonoverlapping

python3 create_node_propensity.py --dataset ciao --inDirectory louvain --outDirectory propensity_subgraph_louvain --path_to_ngraph ngraph_centrality

python3 find_center_of_communities.py --dataset ciao --directory propensity_subgraph_louvain

python3 trust_prediction.py --dataset ciao --graphfile renumbered_graph_ciao.csv --communityfile louvain_ciao.csv --community_center propensity_subgraph_louvain/centerclusters.csv --ratingfile rating.csv --overlap nonoverlap

