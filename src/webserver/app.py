from flask import Flask
from flask import jsonify
import networkx as nx
from datetime import datetime


app = Flask(__name__)
G = nx.Graph()

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
    #TODO should return the id of vertex_id in community structure community_name
    pass

@app.route('/community/<community_name>/all/<int:community_id>')
def community_all(community_name:str, community_id:int):
    #TODO should return the ids of all vertices in community_id from community structure community_name
    pass


@app.route('/')
def index():
    return 'Web App with Python Flask!'




#load_graph('data/dblp-coauthor.edgelist')
load_graph('data/sample_HCI_coauthornet.edgelist')

app.run(host='0.0.0.0')
