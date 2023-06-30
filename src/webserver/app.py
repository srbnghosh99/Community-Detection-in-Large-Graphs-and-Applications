from flask import Flask
from flask import jsonify
import networkx as nx
from datetime import datetime
import csv
from flask import Flask,render_template,request
import json

app = Flask(__name__)
G = nx.Graph()

nonoverlappingcommunity_vertexmap={}  #nonoverlappingcommunity_vertexmap[cdalgoname:str][vertexid:str]:int
nonoverlappingcommunity_communitymap={} #nonoverlappingcommunity_vertexmap[cdalgoname:str][communityid:int]:list:str
overlappingcommunity_vertexmap={}  #nonoverlappingcommunity_vertexmap[cdalgoname:str][vertexid:str]:int
overlappingcommunity_communitymap={} #nonoverlappingcommunity_vertexmap[cdalgoname:str][communityid:int]:list:str


def load_graph(edgelist_filename: str):
    global G
    print("Starting to load graph at =", datetime.now().strftime("%H:%M:%S"))
    G = nx.read_edgelist(edgelist_filename, delimiter=" ", data=(("weight", int),))
    print("Finished loading graph at =", datetime.now().strftime("%H:%M:%S"))
    print()

# returns a JSON where ["data"] is a list of  vertexid:str
@app.route('/vertices')
def vert():
    ret={}
    ret["data"] = list(G.nodes)
    ret["status"] = "OK"
    return jsonify(ret)

# returns a JSON where ["data"] is a list of neighbors:str of that vertex
@app.route('/neighborsof/<vertex_id>')
def neighbor(vertex_id: str):
    ret = {}
    ret["status"] = "OK"
    ret["data"] = list (G.neighbors(vertex_id))
    return jsonify(ret)


def adj_list(graph):
    adj_list = {}
    for a in list(graph.nodes):
        adj_list [a] = list (graph.neighbors(a))
    return adj_list

@app.route('/wholeadjlist')
def wholeadjlist():
    ret = {}
    ret ["status"]="OK"
    ret["data"] = adj_list(G)
    return jsonify(ret)


#returns a JSON where ["data"][vertexid] is a communityid:int
@app.route('/community/<communityalgo_name>/vertex/<vertex_id>')
def community_of(communityalgo_name:str, vertex_id: str):
    ret = {}
    data = {}
    community = {}
    try:
        data[vertex_id] = nonoverlappingcommunity_vertexmap[communityalgo_name][vertex_id]
        ret["status"] = "OK"
        ret["data"] = data
        
    except:
        ret["status"] = "KO"
    
    return jsonify(ret)
    

#returns a JSON where ["data"][community_id] is a list of vertexid:str belonging to that community
@app.route('/vertex/<vertex_id>')
def community_all_for_vertex(vertex_id:str):
    ret={}
    overlap_comm = {}
    try:
        ret["Name"] = vertex_id
        communities_instore1 = nonoverlappingcommunity_communitymap.keys()
        lis1 = list(communities_instore1)
        for i in lis1:
            ret[i] =  nonoverlappingcommunity_vertexmap[i][vertex_id]
        communities_instore2 = overlappingcommunity_communitymap.keys()
        lis2 = list(communities_instore2)
        for i in lis2:
            ret[i] =  overlappingcommunity_vertexmap[i][vertex_id]
            overlap_comm[i]=overlappingcommunity_vertexmap[i][vertex_id]
            ret['Degree'] = G.degree[vertex_id]

        comm_vertices = []
        for algo_name, comm_id in overlap_comm.items():
            for i in comm_id:
                lis = list(overlappingcommunity_communitymap[algo_name][i])
                comm_vertices.append(lis)
            flat_list = [item for sublist in comm_vertices for item in sublist]
            ret[algo_name] = flat_list
        ret["status"] = "OK"
    except:
        ret["status"] = "KO"

    return jsonify(ret)

#returns a JSON where ["data"][community_id] is a list of vertexid:str belonging to that community
@app.route('/community/<communityalgo_name>/all/<int:community_id>')
def community_all(communityalgo_name:str, community_id:int):
    ret={}

    try:
        communityset = nonoverlappingcommunity_communitymap[communityalgo_name][community_id]
        ret["data"] = {}
        ret["data"][community_id] = list(communityset) #set() are not JSON serializable in python
        
        ret["status"] = "OK"
    except:
        ret["status"] = "KO"

    return jsonify(ret)

#returns a JSON where data is a list of strings of available communities
@app.route('/communities')
def communities():
    ret = {}
    try:
        comm_info = {}
        communities_instore1 = nonoverlappingcommunity_communitymap.keys()
        lis1 = list(communities_instore1)
        for i in lis1:
            comm_info[i] =  len(nonoverlappingcommunity_communitymap[i])
        communities_instore2 = overlappingcommunity_communitymap.keys()
        lis2 = list(communities_instore2)
        for i in lis2:
            comm_info[i] =  len(overlappingcommunity_communitymap[i])
        ret["data"] = comm_info
        ret["status"] = "OK"
    except:
        ret["status"] = "KO"
    return jsonify(ret)
        
@app.route('/')
def index():
    return 'Web App with Python Flask!'

@app.route('/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('data.html',form_data = form_data)

def build_nonoverlappingcommunitymap_fromvertexmap(communityalgo_name:str):
    print(str)
    reversemap = {}
    vertexmap = nonoverlappingcommunity_vertexmap[communityalgo_name]
    for vertexid in vertexmap:
        commid = vertexmap[vertexid]
        if commid not in reversemap:
            reversemap[commid] = set()
        reversemap[commid].add(vertexid)
    nonoverlappingcommunity_communitymap[communityalgo_name]=reversemap

def build_overlappingcommunitymap_fromvertexmap(communityalgo_name:str):
    print(str)
    reversemap = {}
    vertexmap = overlappingcommunity_vertexmap[communityalgo_name]
    for vertexid in vertexmap:
        commid = vertexmap[vertexid]
        for i in commid:
            if i not in reversemap:
                reversemap[i] = set()
            reversemap[i].add(vertexid)
    overlappingcommunity_communitymap[communityalgo_name]=reversemap


# loads a community file. Assume that the format of the file is:
#
# with a one line header
# Vertex Community
# vertexid:str communityid:int
def load_community_nonoverlapping(communityalgo_name:str, filename:str):
    comm = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        for row in reader:
            comm[row['Vertex']] = int(row['Community'])
        nonoverlappingcommunity_vertexmap[communityalgo_name] = comm
        build_nonoverlappingcommunitymap_fromvertexmap(communityalgo_name)

def load_community_overlapping(communityalgo_name:str, filename:str):
    comm = {}
    print(filename)
    with open(filename, 'r') as f:
        comm = json.load(f)
        overlappingcommunity_vertexmap[communityalgo_name] = comm
        build_overlappingcommunitymap_fromvertexmap(communityalgo_name)


#load_graph('data/dblp-coauthor.edgelist')
load_graph('data/sample_HCI_coauthornet.edgelist')
load_community_nonoverlapping('Louvain', 'data/louvain_HCI.csv')
load_community_overlapping('EgoSplitting', 'data/Egosplitting_HCI_memberships.json')

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(host='0.0.0.0', port=8080)
