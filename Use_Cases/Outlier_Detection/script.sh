python3 auxiliary_comm.py --graph_file YelpHotel/graph_edges.csv --community_file YelpHotel/spectral.csv --outputfile YelpHotel/spectral_auxiliary_communities1.csv

python3 feature_extraction.py --graph_file YelpHotel/graph_edges.csv --auxiliary_community_file YelpHotel/spectral_auxiliary_communities1.csv --outputfile YelpHotel/spectral_graph_features2.csv

python3 prediction.py --graph_features YelpHotel/spectral_graph_features2.csv --labelled_file YelpHotel/YelpHotel_labels.csv
