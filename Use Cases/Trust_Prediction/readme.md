## Run Commands

python3 subgraph_create.py --graphfile /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/renumbered_graph_epinions.csv --inputfilename /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/louvain_epinions_trustnet.csv --overlapping nonoverlapping --outdirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/louvain


python3 node_propensity.py --inDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/louvain --outDirectory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/propensity_subgraph_louvain


python3 find_center_of_communities.py --directory /Users/shrabanighosh/Downloads/data/trust_prediction/epinions/propensity_subgraph_louvain

python3 trust_prediction.py --graphfile /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/renumbered_graph_ciao.csv --communityfile /Users/shrabanighosh/Downloads/data/trust_prediction/community_clusters/spectral_ciao_c10.csv --community_center /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/propensity_subgraph_spectral/centerclusters.csv --ratingfile /Users/shrabanighosh/Downloads/data/trust_prediction/ciao/ciao_rating.csv --overlap nonoverlap
