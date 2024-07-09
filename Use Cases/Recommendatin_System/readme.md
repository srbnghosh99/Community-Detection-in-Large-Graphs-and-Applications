## Run Commands

###  Download the mat files from this link --> "https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm"

### Process downloaded rawfile 
python3 process_rawfile.py --dataset epinions --directory /Users/shrabanighosh/Downloads/data/recommendation_system


### Run different community detection algorithms for overlapping communities, convert json file to csv and process the 
python3  jsontocsv.py --inputfilename /recommendation_system/ciao/ego_splitting_res2.5_min10.json --outputfilename recommendation_system/ciao/ego_splitting_res2.5_min10.csv

python3 overlapping_processing.py --inputfilename /recommendation_system/ciao/ego_splitting_res2.5_min10.csv

### Create community json files

python3 social_recommendation_system.py --graph /recommendation_system/epinions/renumbered_graph_epinions.csv --cdfile /recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --outdir /recommendation_system/epinions/ego_splitting_epinions_res2.5_min5/ --overlap overlapping

### Create the node propensity for each community in json format

python3 /trust_prediction/node_propensity.py --inDirectory /recommendation_system/epinions/ego_splitting_epinions_res2.5_min5 --outDirectory /recommendation_system/epinions/propensity_res2.5_min5

### Convert the json file in csv format

python3 node_propensity.py --cdfile /recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /recommendation_system/epinions/propensity_res2.5_min5 --outputdir /recommendation_system/epinions/propensity_res2.5_min5 --overlap overlapping

### Compute rating prediction
python3 rating_prediction.py --dataset epinions --trustnet /recommendation_system/epinions/renumbered_graph_epinions.csv --ratingfile /recommendation_system/epinions/epinions_rating.csv --communityfile /recommendation_system/epinions/ego_splitting_epinions_res2.5_min5.csv --inputdir /recommendation_system/epinions --output_dir /recommendation_system/epinions/propensity_res2.5_min5/ --overlap overlapping
