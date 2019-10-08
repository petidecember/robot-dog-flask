
let avatar = document.getElementById("avatar");
let avatar1 = document.getElementById("avatar1");
let main = document.getElementById("main");
main.style.display='none';
main.style.opacity="0";
let streamm = document.getElementById("stream");
let camerset = document.getElementById("camera");
let controls = document.getElementById("controls");
let controlsp = document.getElementById("controlsphone");
function redir(url){
    setTimeout(function(){
        window.open(url, "_blank");
    },
    2000);
}

streamm.addEventListener("mouseover",function(){
    camerset.style.display="none";
    controls.style.display="none";
    streamm.style.width="95%";
    streamm.style.borderRadius="30px 30px";
    streamm.style.border="1px solid white";
    streamm.style.boxShadow="0px 0px 50px white";
    streamm.style.marginBottom="15vh";
    controlsp.style.display="flex";
});

streamm.addEventListener("mouseout",function(){
    camerset.style.display="flex";
    controls.style.display="flex";
    streamm.style.width="50%";
    streamm.style.borderRadius="0";
    streamm.style.border="none";
    streamm.style.boxShadow="none";
    streamm.style.marginBottom="20px";
    controlsp.style.display="none";
});

controlsp.addEventListener("mouseover",function(){
    camerset.style.display="none";
    controls.style.display="none";
    streamm.style.width="95%";
    streamm.style.borderRadius="30px 30px";
    streamm.style.border="1px solid white";
    streamm.style.boxShadow="0px 0px 50px white";
    streamm.style.marginBottom="6vh";
    controlsp.style.display="flex";
});

controlsp.addEventListener("mouseout",function(){
    camerset.style.display="flex";
    controls.style.display="flex";
    streamm.style.width="50%";
    streamm.style.borderRadius="0";
    streamm.style.border="none";
    streamm.style.boxShadow="none";
    streamm.style.marginBottom="20px";
    controlsp.style.display="none";
});

main.addEventListener("mouseover",function(){

});
avatar.addEventListener("click",function(){
    //redir("../index1.html");
    setTimeout(start,2500);
    setTimeout(function(){
        main.style.display="block";
        main.style.opacity="1";
     //   $("#main").animate({opacity: '1', borderRadius:30});
        avatar.style.display="none";
        avatar1.style.display="none";
        //$("#avatar").animate({opacity: '0'});
       // $("#avatar1").animate({opacity: '0'});
    
    }, 2000);
})
//https://s3-us-west-2.amazonaws.com/s.cdpn.io/499416/demo-bg.jpg

