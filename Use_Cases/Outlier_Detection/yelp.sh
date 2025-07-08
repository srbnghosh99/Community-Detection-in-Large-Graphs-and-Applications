#!/bin/bash

START_IDX=$1
END_IDX=$2

echo "Running from $START_IDX to $END_IDX"

start_time=$(date +%s)


########################### modularity  ###########################
#python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file YelpHotel/input_data/selected_solutions_2_output_random5.bson --outputfile YelpHotel/prediction_output/selected_solutions_mod_0_output_random12 --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX

#python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file YelpHotel/input_data/selected_solutions_modularity_0_random13.bson --outputfile YelpHotel/prediction_output/selected_solutions_mod_0_output_random13 --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX

#python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file /users/sghosh15/Genetic_algo/output_yelp/output_random13/modularity_output_random13_features.csv --outputfile /users/sghosh15/Genetic_algo/output_yelp/output_random13/prediction_anomaly --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX
########################### conductance ###########################
#python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file /users/sghosh15/Genetic_algo/output_yelp/random_conductance_output4/random_conductance_output4_features.csv --outputfile /users/sghosh15/Genetic_algo/output_yelp/random_conductance_output4/prediction_anomaly --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX

########################### density ###########################
#python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file /users/sghosh15/Genetic_algo/output_yelp/random_density_output5/features/random_density_output5_features.csv --outputfile /users/sghosh15/Genetic_algo/output_yelp/random_density_output5/prediction_anomaly --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX

########################### cc ########################

python3 codes/auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file /users/sghosh15/Genetic_algo/output_yelp/random_cc_output4/features/selected_solutions_clustering_coeff_0/merged.csv --outputfile /users/sghosh15/Genetic_algo/output_yelp/random_cc_output4/prediction_anomaly --labelled_file YelpHotel/YelpHotel_labels.csv  --start_index $START_IDX --end_index $END_IDX

#python3 auxiliary_comm.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --community_file YelpHotel/spectral.csv --outputfile YelpHotel/spectral_auxiliary_communities1.csv

#python3 feature_extraction.py --graph_file YelpHotel/yelp_hotel_graph_edges.csv --auxiliary_community_file YelpHotel/spectral_auxiliary_communities1.csv --outputfile YelpHotel/spectral_graph_features2.csv

#python3 prediction.py --graph_features YelpHotel/spectral_graph_features2.csv --labelled_file YelpHotel/YelpHotel_labels.csv

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
# Calculate the difference in minutes
time_diff_minutes=$((elapsed_time / 60))

# Print the result
echo "Time elapsed: $time_diff_minutes minutes"

# Print the result including seconds
echo "Time elapsed: $time_diff_minutes minutes and $((time_diff_seconds % 60)) seconds"



