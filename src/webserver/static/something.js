
function test() {
    console.log("hello");
    const myp = document.querySelector("p");
    myp.textContent = "Hello world!";
}

window.addEventListener("load", test);
