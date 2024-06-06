## Run Commands

python3 social_recommendation_system.py --graph /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/renumbered_graph_epinions.csv --cdfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --outdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5/ --overlap overlapping

python3 /Users/shrabanighosh/Downloads/data/trust_prediction/node_propensity.py --inDirectory /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5 --outDirectory /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5


python3 node_propensity.py --cdfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5 --outputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5 --overlap overlapping

python3 rating_prediction.py --dataset epinions --trustnet /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/renumbered_graph_epinions.csv --ratingfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/epinions_rating.csv --communityfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions --output_dir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5/ --overlap overlapping
