import {add_d3_graph} from "/static/d3render.js";
import {postJSON} from "util";

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
	//aborted implementation becasue falsk does it well on its own
	
	return adjlist;
    }
    
    return loading();
}

function ego_display(vertexid, domstring) {
    ego (vertexid)
    	.then((item) => from_adjlist_to_d3structure(item))
	.then((d3data) =>{
//	    console.log(JSON.stringify(d3data));
	    add_d3_graph(d3data, domstring);
	});
}

//communityalgoname is a string
//vertices is a list of vertexid:str
function get_community_data (communityalgoname, vertices) {
    console.log("get_community_data");
    const data = { 'communityalgoname': communityalgoname,
		   'vertexids': vertices };
    postJSON('/communitybatch', data).then(i => console.log(i));
}


export {
    ego_display,
    get_community_data
};
