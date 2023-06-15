from flask import Flask
import networkx as nx
from datetime import datetime

app = Flask(__name__)
G = nx.Graph()

def load_graph(edgelist_filename: str):
    print("Starting to load graph at =", datetime.now().strftime("%H:%M:%S"))
    G = nx.read_edgelist(edgelist_filename, delimiter=" ", data=(("weight", int),))
    print("Finished loading graph at =", datetime.now().strftime("%H:%M:%S"))

@app.route('/vertices')
def vert():
    #TODO should return a JSON list of all vertices
    ret=G.nodes
    return jsonify(ret)

@app.route('/neighborsof/<id>')
def neighbor(id: str):
    #TODO should return a JSON list of all neighbors of that vertex
    pass

@app.route('/')
def index():
    return 'Web App with Python Flask!'



#load_graph('data/dblp-coauthor.edgelist')
load_graph('data/sample_HCI_coauthornet.edgelist')
app.run(host='0.0.0.0')
