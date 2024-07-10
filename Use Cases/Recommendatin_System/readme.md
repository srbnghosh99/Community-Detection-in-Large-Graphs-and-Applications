## Run Commands

###  Download the mat files from this link --> "https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm"

### Process downloaded rawfile 
python3 process_rawfile.py --dataset ciao --directory path_to_rawfile


### Run different community detection algorithms for overlapping communities, convert json file to csv and process the 
python3  jsontocsv.py --inputfilename /recommendation_system/ciao/ego_splitting_res2.5_min10.json --outputfilename recommendation_system/ciao/ego_splitting_res2.5_min10.csv

python3 overlapping_processing.py --inputfilename /recommendation_system/ciao/ego_splitting_res2.5_min10.csv

### Create community json files

python3 social_recommendation_system.py --graph /recommendation_system/ciao/renumbered_graph_epinions.csv --cdfile /recommendation_system/ciao/ego_splitting_res2.5_min10.csv --outdir /recommendation_system/ciao/ego_splitting_res2.5_min10/ --overlap overlapping

### Create the node propensity for each community in json format
### To run this python file, this module "https://github.com/srbnghosh99/Community-Detection-in-Large-Graphs-and-Applications-ngraph.centrality" is required.

python3 /trust_prediction/node_propensity.py --inDirectory /recommendation_system/ciao/ego_splitting_res2.5_min10 --outDirectory /recommendation_system/ciao/propensity_res2.5_min10

### Convert the json file in csv format

python3 node_propensity.py --cdfile /recommendation_system/ciao/ego_splitting_res2.5_min5.csv --inputdir /recommendation_system/ciao/propensity_res2.5_min10 --outputdir /recommendation_system/ciao/propensity_res2.5_min10 --overlap overlapping

### Compute rating prediction
python3 rating_prediction.py --dataset ciao --trustnet /recommendation_system/ciao/renumbered_graph_ciao.csv --ratingfile /recommendation_system/ciao/ciao_rating.csv --communityfile /recommendation_system/ciao/ego_splitting_res2.5_min10.csv --inputdir /recommendation_system/ciao --output_dir /recommendation_system/ciao/propensity_res2.5_min10/ --overlap overlapping
