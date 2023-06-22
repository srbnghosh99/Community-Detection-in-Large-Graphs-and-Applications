from flask import Flask
from flask import jsonify
import networkx as nx
from datetime import datetime
import csv

app = Flask(__name__)
G = nx.Graph()

nonoverlappingcommunity_vertexmap={}  #nonoverlappingcommunity_vertexmap["community"]["vertexid"]:int
nonoverlappingcommunity_communitymap={} #nonoverlappingcommunity_vertexmap["community"][communityid]:list:str


def load_graph(edgelist_filename: str):
    global G
    print("Starting to load graph at =", datetime.now().strftime("%H:%M:%S"))
    G = nx.read_edgelist(edgelist_filename, delimiter=" ", data=(("weight", int),))
    print("Finished loading graph at =", datetime.now().strftime("%H:%M:%S"))

@app.route('/vertices')
def vert():
    ret=list(G.nodes)
    return jsonify(ret)

# returns a JSON list of all neighbors of that vertex
@app.route('/neighborsof/<vertex_id>')
def neighbor(vertex_id: str):
    ret = list (G.neighbors(vertex_id))
    return jsonify(ret)

@app.route('/community/<community_name>/vertex/<vertex_id>')
def community_of(community_name:str, vertex_id: str):
    #TODO should
    ret = {}
    data = {}

    try:
        data[vertex_id] = nonoverlappingcommunity_vertexmap[community_name][vertex_id]
        ret["status"] = "OK"
        ret["data"] = data
    except:
        ret["status"] = "KO"
    
    return jsonify(ret)

@app.route('/community/<community_name>/all/<int:community_id>')
def community_all(community_name:str, community_id:int):
    ret={}

    try:
        communityset = nonoverlappingcommunity_communitymap[community_name][community_id]
        ret["data"] = list(communityset) #set() are not JSON serializable in python
        ret["status"] = "OK"
    except:
        ret["status"] = "KO"

    return jsonify(ret)

@app.route('/communities')
def communities():
    ret = {}
    try:
        ret["data"] = list(nonoverlappingcommunity_communitymap.keys())
        ret["status"] = "OK"
    except:
        ret["status"] = "KO"
    return jsonify(ret)
        
@app.route('/')
def index():
    return 'Web App with Python Flask!'

def build_nonoverlappingcommunitymap_fromvertexmap(commname:str):
    reversemap = {}
    vertexmap = nonoverlappingcommunity_vertexmap[commname]
    for vertexid in vertexmap:
        commid = vertexmap[vertexid]
        if commid not in reversemap:
            reversemap[commid] = set()
        reversemap[commid].add(vertexid)
    nonoverlappingcommunity_communitymap[commname]=reversemap


# loads a community file. Assume that the format of the file is:
#
# with a one line header
# Vertex Community
# vertexid:str communityid:int
def load_community_nonoverlapping(commname:str, filename:str):
    comm = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        for row in reader:
            comm[row['Vertex']] = int(row['Community'])
        nonoverlappingcommunity_vertexmap[commname] = comm
        build_nonoverlappingcommunitymap_fromvertexmap(commname)


#load_graph('data/dblp-coauthor.edgelist')
load_graph('data/sample_HCI_coauthornet.edgelist')
load_community_nonoverlapping('Louvain', 'data/louvain_HCI.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
