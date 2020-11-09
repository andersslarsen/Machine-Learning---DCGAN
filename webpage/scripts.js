function myFunction() {
  document.getElementById("ja").innerHTML = "Riktig, an fins!";
}

function myFunction2() {
  document.getElementById("nei").innerHTML = "Riktig, an fins kje!";
}

(() => {
  setTimeout(() => {
    document.getElementsByTagName("body")[0].style.backgroundColor = 'lightgreen';
  }, 300)
})();

function sampleFunction(){
  location.reload(true);
}


function randomNum(){
  return Math.floor(Math.random() * 10) + 10;
}


function newImage(){
  var img = new Image();
  var num = randomNum();
  img.src="static/images/0000" + num + ".jpg";
  //img.src = "static/images/000018.jpg";
  document.getElementById('body').appendChild(img);
  console.log(img)
}
