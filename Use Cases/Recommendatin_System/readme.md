## Run Commands

### Process downloaded rawfile 
python3 process_rawfile.py --dataset epinions --directory /Users/shrabanighosh/Downloads/data/recommendation_system


### Run different community detection algorithms for overlapping communities, convert json file to csv and process the 
python3  jsontocsv.py --inputfilename /Users/shrabanighosh/Downloads/data/recommendation_system/ciao/ego_splitting_res2.5_min10.json --outputfilename /Users/shrabanighosh/Downloads/data/recommendation_system/ciao/ego_splitting_res2.5_min10.csv

python3 overlapping_processing.py --inputfilename /Users/shrabanighosh/Downloads/data/recommendation_system/ciao/ego_splitting_res2.5_min10.csv

### Create community json files

python3 social_recommendation_system.py --graph /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/renumbered_graph_epinions.csv --cdfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --outdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5/ --overlap overlapping

### Create the node propensity for each community in json format

python3 /Users/shrabanighosh/Downloads/data/trust_prediction/node_propensity.py --inDirectory /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5 --outDirectory /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5

### Convert the json file in csv format

python3 node_propensity.py --cdfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5 --outputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5 --overlap overlapping

### Compute rating prediction
python3 rating_prediction.py --dataset epinions --trustnet /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/renumbered_graph_epinions.csv --ratingfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/epinions_rating.csv --communityfile /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions --output_dir /Users/shrabanighosh/Downloads/data/recommendation_system/epinions/propensity_res2.5_min5/ --overlap overlapping
