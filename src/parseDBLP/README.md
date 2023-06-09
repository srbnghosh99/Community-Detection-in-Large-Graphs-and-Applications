# Overview

The codes in this directory are used to extract a DBLP coauthorship graph from DBLP's XML

the coauthor_net_generate.py file parses a DBLP CSV into a graph in the edgelist format

## how to run the code?

To convert xml files in a directory to csv files run the code parsedblp.py 

The command line to run the file: 

python3 parsedblp.py  --inputdirectory data/xmlfiles --outputdirectory data/csvfiles 

The command line to merge all csv files:

python3 merge_csvfiles.py

To convert csv files to authorship network

The command line to run the file: 

python3 coauthor_net_generate.py  --filename fname

The command line to run author name mapping to identical id:

python3 author_nameid_map.py --inputfile merge_file.edgelist --outputfile merge_file_idname.edgelist



