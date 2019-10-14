let x, y;
let joystickphone = document.getElementById("joystickbox");
let joystickbuttn = document.getElementById("joystickbutton");
joystickbuttn.style.transform="translate(0px, 0px)";

function polToCar(a,r, maxr,z){
    if(r>maxr){
        r=maxr;
    }
    let x = Math.sin(a)*r;
    let y = Math.cos(a)*r;

    if(z){
        return x;
    }
    else{
        return y;
    }
}


function carToPol(x,y,z){
    let r = Math.sqrt(x*x+y*y);
    let a;
        a = Math.atan(x/y);
    if(y<0){
        a = -Math.atan(x/y);
    }

    if(z){
        return a;
    }
    else{
        return r;
    }
}


joystickphone.addEventListener("touchstart",posn);
function posn(event){
    x = event.touches[0].clientX;
    y = event.touches[0].clientY;
}

let devider=0;
let sendIntrv;
let xjoy;
let yjoy;
let r;
let angl;
let tur;
joystickphone.addEventListener("touchmove",jsjoy);
function jsjoy(event) {
    event.preventDefault();
    let xn = event.touches[0].clientX;
    let yn = event.touches[0].clientY;
    let xx = xn-x;
    let yy = yn-y;
     xjoy = polToCar(carToPol(xx,yy,true),carToPol(xx,yy,false),100,true);
     yjoy = polToCar(carToPol(xx,yy,true),carToPol(xx,yy,false),100,false);
     r = carToPol(xjoy, yjoy, false);
     tur = carToPol(xjoy, yjoy, true);
    if(yy<0){
        yjoy=-yjoy;
    }
    joystickbuttn.style.transform="translate("+xjoy+"px, "+yjoy+"px)";
    document.getElementById("demo").innerHTML=r+", "+xjoy;
    if(xjoy<1&&xjoy>-1){
        xjoy=1;
    }
    if(yy<0){
        angl=90;
    }
    else{
        angl=270;
    }
    standing = false;
    ws.send('joyL;'+angl+';'+parseInt(r));
    ws.send('joyR;'+(parseInt(xjoy)));
    let cam = 90+parseInt(-20*(xjoy/100));
    ws.send('setC;'+cam);
}

joystickphone.addEventListener("touchend",posres);
function posres(event){
    xjoy=0;
    r=0;
    joystickbuttn.style.transform="translate(0px,0px)";
    document.getElementById("demo").innerHTML="0,0";
    ws.send('joyR;0');
    ws.send('setC;90');
    ws.send('joyL;90;0');
    standing = true;
}
