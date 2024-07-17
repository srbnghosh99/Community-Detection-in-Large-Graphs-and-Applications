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

python3 process_rawfile.py --dataset ciao

python3 ../jsontocsv.py --dataset ciao --inputfilename ego_splitting_res2.5_min10.json

##python3 overlapping_processing.py --dataset ciao --inputfilename ego_splitting_res2.5_min10.csv

python3 social_recommendation_system.py --dataset ciao --graphfile renumbered_graph_ciao.csv --cdfile  ego_splitting_res2.5_min10.csv --outdir ego_splitting_res2.5_min10/ --overlap overlapping

python3 ../trust_prediction/create_node_propensity.py --dataset ciao --inDirectory ego_splitting_res2.5_min10 --outDirectory propensity_res2.5_min10

python3 generate_dataframe_from_propensity.py --dataset ciao --cdfile ego_splitting_res2.5_min10.csv --inputdir propensity_res2.5_min10/ --outputdir propensity_res2.5_min10 --overlap overlapping
#
python3 rating_prediction.py --dataset ciao --trustnet renumbered_graph_ciao.csv --ratingfile rating.csv --communityfile ego_splitting_res2.5_min10.csv --inputdir /recommendation_system/ciao --output_dir propensity_res2.5_min10/ --overlap overlapping
