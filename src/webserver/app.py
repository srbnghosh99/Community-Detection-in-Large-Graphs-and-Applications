from flask import Flask
from flask import jsonify
import networkx as nx
from datetime import datetime
import csv
from flask import Flask,render_template,request
import json
import pandas as pd

app = Flask(__name__)
G = nx.Graph()
nonoverlappingcommunity_vertexmap={}  #nonoverlappingcommunity_vertexmap[cdlgoname:str][vertexid:str]:int
nonoverlappingcommunity_communitymap={} #nonoverlappingcommunity_vertexmap[cdalgoname:str][communityid:int]:list:str
overlappingcommunity_vertexmap={}  #nonoverlappingcommunity_vertexmap[cdlgoname:str][vertexid:str]:int
overlappingcommunity_communitymap={} #nonoverlappingcommunity_vertexmap[cdalgoname:str][communityid:int]:list:str


def load_graph(edgelist_filename: str):
    global G
    print("Starting to load graph at =", datetime.now().strftime("%H:%M:%S"))
    G = nx.read_edgelist(edgelist_filename, delimiter=" ", data=(("Weight", int),))
    attri_dict = (sorted(G.degree, key=lambda x: x[1], reverse=True))
    nx.set_node_attributes(G, attri_dict, name="degree")
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

def adj_list(graph):
    adj_list = {}
    for a in list(graph.nodes):
        adj_list [a] = list (graph.neighbors(a))
    return adj_list

#return a JSON such that ["data"] is an adjacency list of the whole graph.
#That is to say such that ["data"][vertexId:str] is a list of vertex id:str
@app.route('/wholegraph')
def wholeadjlist():
    ret = {}
    ret ["status"]="OK"
    ret["data"] = adj_list(G)
    return jsonify(ret)

#return a JSON such that ["data"] is an adjacency list of the ego network of vertex_id.
# That is to say such that ["data"][vertexId:str] is a list of vertex id:str
# @app.route('/ego/<vertex_id>')
# def egoadjlist(vertex_id:str):
#     # ret["Name"] = vertex_id
#     print("Inside ego vertex")
#     ret = {}
#     ret["Name"] = vertex_id
#     l = list(G.neighbors(vertex_id))
#     # ret["l"] = l

    
#     # ret["comm"] = overlappingcommunity_vertexmap['Louvain'][vertex_id]
#     l.append(vertex_id)
#     egonet = G.subgraph(l)
#     ret["data"] = adj_list(egonet)
#     comm = {}
#     # comm["T_Gedeon"]['c'] = nonoverlappingcommunity_vertexmap['Louvain'][i]
#     for i in l:
#         comm[i] = nonoverlappingcommunity_vertexmap['Louvain'][i]
#         # comm[i] = overlappingcommunity_vertexmap['EgoSplitting'][i]
#     ret["group"] = comm
#     ret ["status"]="OK"
#     return jsonify(ret)


@app.route('/hist')
def influential():
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    dict = (sorted(G.degree, key=lambda x: x[1], reverse=True))
    degree_df = pd.DataFrame.from_dict(dict)
    degree_df.columns = ['Node', 'Degree']
    degree_df= degree_df[0:31]
    ret = {}
    l = degree_df["Node"].tolist()
    ret ["status"]="OK"
    ret["data"]=l
    #how to run histogram html
    return jsonify(ret)

@app.route('/distance/<vertex_id1>/<vertex_id2>')
def shortestpath(vertex_id1:str, vertex_id2:str):
    ret = {}
    l = nx.shortest_path(G, vertex_id1, vertex_id2)
    ret ["status"]="OK"
    ret["data"]= l
    return jsonify(ret)


@app.route('/ego/<vertex_id>')
def egoadjlist(vertex_id:str):
    ret = {}
    ret ["status"]="OK"
    l = list(G.neighbors(vertex_id))
    l.append(vertex_id)
    egonet = G.subgraph(l)
    ret["data"] = adj_list(egonet)
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

# @app.route('/communities/compare/<int:vertex_id>')
# def compare_non_overlapping(vertex_id:str):
#     ret = {}
#     try: 

        
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


#
# expect a POST request formated {'communityalgoname': communityalgoname:str, 'vertexids': [vertexid:str]}
#
@app.route('/communitybatch', methods=['POST'])
def communitybatch():
    ret={}
    try:
        myjson = request.get_json(force=True)
        algoname = myjson["communityalgoname"]
        vertexids = myjson['vertexids']
        print(algoname, vertexids)
        commdict = {}
        for vid in vertexids:
            try :
                commdict[vid] = nonoverlappingcommunity_vertexmap[algoname][vid]
            except:
                pass #it is expected that some vertices are not in a community
            ret["data"] = commdict
        ret["status"] = "OK"
    except:
        ret = {}
        ret["status"] = "KO"
    
    return jsonify(ret)

    
        
# load_graph('dblp-coauthor.edgelist')
load_graph('data/sample_HCI_coauthornet.edgelist')
load_community_nonoverlapping('Louvain', 'data/louvain_HCI.csv')
# load_community_nonoverlapping('Deepwalk', 'assignments/deepwalk_walk1_name.csv')
# load_community_nonoverlapping('GEMSEC', 'assignments/gemesec_walk1_name.csv')
load_community_overlapping('EgoSplitting', 'data/Egosplitting_HCI_memberships.json')

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(host='0.0.0.0', port=8080)
