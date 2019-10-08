let ws;
let streamWs;
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
        ws.send('joyR;180;100');
        ws.send('setC;110');
    }
    else if(map[68]||map[39]) { //d
        ws.send('joyR;0;100');
        ws.send('setC;70');
    }

}

function stopMove(event) {
    if(event.keyCode == 65 || event.keyCode == 68 || event.keyCode == 37 || event.keyCode == 39) {
        ws.send('joyR;0;0');
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

function follow_ball() {
    ball_follow = !ball_follow;
    if(ball_follow) {
        document.getElementById('follow').style.backgroundColor = "lightgreen";
    } else {
        document.getElementById('follow').style.backgroundColor = "darkred";
    }
    ws.send("camera;ball_"+ball_follow);
}

function draw_ball() {
    ball_draw = !ball_draw;
    if(ball_draw) {
        document.getElementById('draw').style.backgroundColor = "lightgreen";
    } else {
        document.getElementById('draw').style.backgroundColor = "darkred";
    }

    ws.send("camera;draw_"+ball_draw);
}

function changeSize() {
    document.getElementById('stream').style.width = document.getElementById('size').value;
}

function changeStabilise() {
    ws.send('camera;' + document.getElementById('method').value);
    console.log('camera;' + document.getElementById('method').value);
}

function start() {
    document.getElementById('start').style.display = 'none';
    ws = new WebSocket("ws://10.3.141.1:1337");
    streamWs = new WebSocket("ws://10.3.141.1:1338");
    ws.onopen = function(event) {
        console.log("open");
    };
    ws.onmessage = function(event) {
        //console.log(event.data);
        let cmd = event.data.split(';');
        if(cmd[0] === "calibration") {
            let r = Number(cmd[1]);
            let g = Number(cmd[2]);
            let b = Number(cmd[3]);
            document.getElementsByClassName('square')[0].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        }
    };
    
    streamWs.onopen = function(event) {
        console.log("open");
    };
    streamWs.onmessage = function(event) {
        var objectURL = URL.createObjectURL(event.data);
        myImage.src = objectURL;
    };
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    ws.close();
    streamWs.close();
    document.getElementById('stop').style.display = 'none';
}
