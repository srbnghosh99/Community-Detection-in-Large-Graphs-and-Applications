# Overview

The models in this directory are existing models for community detection.

The Louvain Algorithm required library from the link to be installed. "https://github.com/taynaud/python-louvain"

The spectral clustering model is from paper "Understanding Regularized Spectral Clustering via Graph Conductance". We did some minor revisions in the code to match with current networkx python library version. The mothodical parts remained unchanged. The code is from this github link. "https://github.com/crisbodnar/regularised-spectral-clustering/tree/master" 


The AMOS spectral clustering algorithm is from paper "AMOS: AN AUTOMATED MODEL ORDER SELECTION ALGORITHM FOR SPECTRAL GRAPH CLUSTERING". Github link "https://github.com/tgensol/AMOS/tree/master". 

EgoSplitting clustering algoritm is from paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters". The code has some minor revision to adjust with current library versions. Methodical sections remained unchanged. Github link "https://github.com/benedekrozemberczki/EgoSplitting". 

DeepWalk clustering algoritm is from paper "DeepWalk: Online Learning of Social Representations". The code has some minor revision to adjust with current library versions. Methodical sections remained unchanged. Github link https://github.com/phanein/deepwalk/tree/master

DMGI clustering algoritm is from paper "Unsupervised Attributed Multiplex Network Embedding ". The code has some minor revision to adjust with current library versions. Methodical sections remained unchanged. Github link https://github.com/pcy1302/DMGI/tree/master

SDCN clustering algoritm is from paper "Structural Deep Clustering Network". The code has some minor revision to adjust with current library versions.  Methodical sections remained unchanged. Github link https://github.com/bdy9527/SDCN

AGE clustering algoritm is from paper "Adaptive Graph Encoder for Attributed Graph Embedding". The code has some minor revision to adjust with current library versions.  Methodical sections remained unchanged. Github link  https://github.com/thunlp/AGE/tree/master


## how to run the code?

Indivually running the models

Spectral Clustering --> python3 spectral_clustering.py --inputfilename filename_directory --outputfilename filename_diectory 

Louvain Algorithm --> python3 louvain_algorithm.py --inputfilename filename_directory --outputfilename filename_diectory 

node2vec -- > 
python3 src/main.py --input graph/karate.edgelist --output emb/karate.emd. ## Graph embedding

EgoSplitting Algorithm --> python3 src/main.py --edge-path input/authorname_edges.edgelist --output-path output/author_cluster_memberships.json

AGE(Adaptive graph encoder) -->

python train.py --dataset cora --gnnlayers 8 --upth_st 0.011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5

SBM-meet_GNN(Adaptive graph encoder) -->

python3 train.py --dataset citeseer --hidden 32_50 --alpha0 10 --split_idx 0 --deep_decoder 1 --model dglfrm_b --epochs 200

GEMSEC -> 
Creating a GEMSEC embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.
$ python src/embedding_clustering.py

Creating a DeepWalk embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.
$ python src/embedding_clustering.py --model DeepWalk

Turning off the model saving.
$ python src/embedding_clustering.py --dump-matrices False

Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.
$ python src/embedding_clustering.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --log-output output/cluster_means/company_means.csv --cluster-mean-output output/logs/company.json

Creating a clustered embedding of the default dataset in 32 dimensions, 20 sequences per source node with length 160 and 10 cluster centers.
$ python src/embedding_clustering.py --dimensions 32 --num-of-walks 20 --random-walk-length 160 --cluster-number 10


