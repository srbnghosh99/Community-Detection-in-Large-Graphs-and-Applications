#!/bin/bash

START_IDX=$1
END_IDX=$2

echo "Running from $START_IDX to $END_IDX"

start_time=$(date +%s)

#python3 codes/subgraph_create.py --dataset ciao --graphfile renumbered_graph_ciao.csv --cdfile /users/sghosh15/Genetic_algo/output_ciao/random_density_output5/random_density_output5_features.csv --subgraphdir sdensity --overlap nonoverlap --flag 2 --ratingfile rating.csv --ground_truthfile ciao_groudtruth.csv  --outputdir /users/sghosh15/Genetic_algo/output_ciao/random_density_output5/output_csvs --start_index $START_IDX --end_index $END_IDX


python3 codes/subgraph_create_prev.py --dataset ciao --graphfile renumbered_graph_ciao.csv --cdfile /users/sghosh15/Genetic_algo/output_ciao/random_density_output5/random_density_output5_features.csv --outdirectory /users/sghosh15/subgraph --overlap nonoverlap --flag 2 --ratingfile rating.csv --ground_truthfile ciao_groudtruth.csv --outputdir /users/sghosh15/Genetic_algo/output_ciao/random_density_output5/output_csvs --start_index $START_IDX --end_index $END_IDX
