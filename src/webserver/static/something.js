import {add_d3_graph} from "/static/d3render.js";

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
	const response = await fetch('/ego/'+vertexid);
	const myjson = await response.json();
	return myjson["data"];
    }
    return loading();
}

//returns a Promise of an adjlist structure
function ego_manual(vertexid) {
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
	//todo: this is incomplete. missed crosslinks between neighbors of vertexid
	
	return adjlist;
    }
    
    return loading();
}



export function test() {
    console.log("test");
    console.log(d3);
    ego ("S_Lucey")
    	.then((item) => from_adjlist_to_d3structure(item))
	.then((d3data) =>{
	    console.log(JSON.stringify(d3data));
	    add_d3_graph(d3data, "#myviz");
	});
}

window.addEventListener("load", test);
