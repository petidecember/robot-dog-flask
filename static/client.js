let ws;
// let streamWs;
let myImage = document.getElementById("stream");

let calibration = false;
let ball_follow = false;
let ball_draw = false;
let standing = true;

myImage.onclick = function() {
    myImage.requestPointerLock();
}

var map = {};
onkeydown = onkeyup = function(e){
    e = e || event;
    map[e.keyCode] = e.type == 'keydown';
}
let id;
document.onpointerlockchange = function() {
    if(document.pointerLockElement === myImage) {
        document.addEventListener('mousemove', moveCamera);
        document.addEventListener('keyup', stopMove);
        id = setInterval(moveRobot, 33);
    } else {
        document.removeEventListener('mousemove', moveCamera);
        document.removeEventListener('keyup', stopMove);
        clearInterval(id);
    }
}
/*
document.getElementById("offset").addEventListener("change", function() {
    console.log(document.getElementById("offset").value);
    ws.send("offset;" + document.getElementById("offset").value);
})
*/


let neut=false;
function moveRobot(event) {
    if(map[87]||map[38]) { //w
        ws.send('joyL;90;100');
        standing = false;
    }
    else if(map[83]||map[40]) { //s
        ws.send('joyL;270;100');
        standing = false;
    }
    else {
        ws.send('joyL;90;0');
        standing = true;
    }

    if(map[65]||map[37]) { //a
        ws.send('joyR;-100');
        ws.send('setC;110');
    }
    else if(map[68]||map[39]) { //d
        ws.send('joyR;100');
        ws.send('setC;70');
    }

}

function stopMove(event) {
    if(event.keyCode == 65 || event.keyCode == 68 || event.keyCode == 37 || event.keyCode == 39) {
        ws.send('joyR;0');
        ws.send('setC;90');
    }
}

function moveCamera(event) {
    if(standing)
        ws.send(`joyC;${event.movementX};${event.movementY}`);
}

function calibrate() {
    if(calibration) {
        calibration = false;
        ws.send("camera;calibrate_done");
        document.getElementById('cal').innerHTML = "Calibrate";
    } else {
        calibration = true;
        ws.send("camera;calibrate");
        document.getElementById('cal').innerHTML = "Apply";
    }
}

function set_follow_ball(set) {
    if(set) {
        document.getElementById('follow').style.backgroundColor = "lightgreen";
    } else {
        document.getElementById('follow').style.backgroundColor = "darkred";
    }
}

function send_toggle_follow() {
    ws.send("camera;toggle_follow");
}

function set_draw_ball(set) {
    if(set) {
        document.getElementById('draw').style.backgroundColor = "lightgreen";
    } else {
        document.getElementById('draw').style.backgroundColor = "darkred";
    }
}

function send_toggle_draw() {
    ws.send("camera;toggle_draw");
}

function changeSize() {
    document.getElementById('stream').style.width = document.getElementById('size').value;
}

function changeStabilise() {
    ws.send('camera;' + document.getElementById('method').value);
    console.log('camera;' + document.getElementById('method').value);
}

function start() {
    ws = new WebSocket("ws://10.3.141.1:1337");
    
    ws.onopen = function(event) {
        console.log("open");
    };
    ws.onmessage = function(event) {
        let cmd = event.data.split(';');
        if(cmd[0] === "calibration") {
            let r = Number(cmd[1]);
            let g = Number(cmd[2]);
            let b = Number(cmd[3]);
            document.getElementsByClassName('square')[0].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        }
        else if(cmd[0] === "camera") {
            
            let set = Boolean(Number(cmd[2]));
            if(cmd[1] === "draw") {
                set_draw_ball(set);
            }
            else if(cmd[1] === "follow") {
                set_follow_ball(set);
            }
        }
        console.log(cmd)
    };
}

start();
