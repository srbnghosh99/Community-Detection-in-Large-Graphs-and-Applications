
//builds a d3 graph structure out of an adjlist
function from_adjlist_to_d3structure(adjlist) {
    let nodes = [];
    let links = [];
    let graph = {"nodes":nodes, "links": links};

    Object.keys(adjlist).forEach((vid) => {
	nodes.push({"id": vid});
	adjlist[vid].forEach((neigh) => {
	    links.push({"source":vid, "target":neigh});
	});
    });

    return graph;
}

//returns a Promise of an adjlist structure
function ego(vertexid) {
    const loading = async() => {
	const response = await fetch('/neighborsof/'+vertexid);
	const myjson = await response.json();

	let adjlist ={};
	adjlist[vertexid] = [];
	myjson["data"].forEach((id)=>{
	    adjlist[id] = [];
	});

	myjson["data"].forEach((id)=>{
	    adjlist[id].push(vertexid);
	    adjlist[vertexid].push(id);
	});
	
	return adjlist;
    }
    
    return loading();
}

function test() {
    ego ("S_Lucey")
    	.then((item) => from_adjlist_to_d3structure(item))
	.then((i) =>{console.log(JSON.stringify(i))});

}

window.addEventListener("load", test);
