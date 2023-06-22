
function test() {
    console.log("hello");
    const myp = document.querySelector("p");
    myp.textContent = "Hello world!";
    const loading = async() => {
	const response = await fetch('/communities');
	const myjson = await response.json();
	myp.textContent = myjson["status"];
    }
    loading();
    
}

window.addEventListener("load", test);
