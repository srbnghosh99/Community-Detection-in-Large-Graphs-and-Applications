# Overview

The codes in this directory are used to extract a DBLP coauthorship graph from DBLP's XML

the coauthor_net_generate.py file parses a DBLP CSV into a graph in the edgelist format

There should be two ways to create a graph file.

1. from the DBLP xml files for particular conferences
2. from the main DBLP xml

## How to get the DBLP conference files?

Go to the conference page on DBLP. for instance:

https://dblp.org/db/conf/d-cscl/index.html

Next to the conference name, hit the download button and select XML.

## How to generate a graph file from individual DBLP conference files?

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


## How to get the DBLP main file.

from here: https://dblp.uni-trier.de/xml/

## How to create the graph of all authors.

There is a DBLPProcess.cpp file to compile.

```make```

should create the binary and

```./DBLPProcess dblp.xml  myout1```

should create the graph in myout1

The code has a dependency to some xml library. read comment at top of cpp file.

## DBLP graph of all authors informaiton

There are total 3352556 number of nodes and 11892185 number of edges.  

Authors collaboration more than 1 time generated graph has 1216501 number of nodes and 2393254 number of edges.  



# Output

The format of the output edgelist file is a standard space separated edge list:

```
u v count
```

where u and v are vertex ids (names like erik_saule) and count is the number of common papers.
