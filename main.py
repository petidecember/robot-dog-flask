from flask import Response
from flask import Flask
from flask import render_template
from pyee import BaseEventEmitter, AsyncIOEventEmitter
from sunfounder import front_wheels, back_wheels
from sunfounder.SunFounder_PCA9685 import Servo
import sunfounder
import threading
from multiprocessing import Process
import cv2
import time
import numpy as np
import websockets
import asyncio
import math

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE,-4) # current camera doesn't support this :(
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if not self.cap.isOpened():
            print("Could not open camera")
            exit()
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            self.grabbed = grabbed
            self.frame = frame

    def read(self):
        #frame = self.frame.copy()
        frame = self.frame # if the program segfaults uncomment the top line and comment out this one
        grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

class ProcessingThread:
    def __init__(self, vs, state):
        self.started = False
        self.vs = vs
        self.state = state
        self.processedFrame = None

    def start(self):
        if self.started:
            print('[!] Frame processing thread has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.process, args=())
        self.thread.start()
        return self

    def process(self):
        vs = self.vs
        someState = self.state

        width, height = someState.resolution()
        resWidth, resHeight = someState.resize_resolution()

        while self.started:
            grabbed, frame = vs.read()
            
            resizedFrame = resize_buffer(frame, resWidth, resHeight)
            
            if someState.calibrate:
                a = resHeight*0.2
                calibBB = calc_sized_bb(resWidth, resHeight, a)
                (x, y, w, h) = round_bb(calibBB)
                cv2.rectangle(resizedFrame, (x, y), (w, h), color=(255, 0, 0), thickness=4)
                
                if someState.processCalibration:
                    someState.processCalibration = False
                    someState.calibrate = False
                    someState.hsv = calibrate(resizedFrame, x, y, w, h)
                    ee.emit('update_colors')
            if someState.followBall:
                success, (x, y, r) = find_ball(frame, someState.hsv, someState.stabilise)
                if success:
                    (x, y, r) = int_circle(scale_circle((x, y, r), (width, height), (resWidth, resHeight)))
                    someState.set_ball_props(x, y, r)
                    ee.emit('ball_found')
            if someState.drawBall:
                (x, y, r) = someState.ball_props()
                cv2.circle(resizedFrame, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(resizedFrame, (x, y), r, (0, 255, 255), 2)

            self.processedFrame = resizedFrame

    def processed_frame(self):
        return self.processedFrame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        pass

class WebsocketThread:
    def __init__(self, state):
        self.started = False
        self.state = state
        self.receiveLoop = asyncio.new_event_loop()
        self.sendLoop = asyncio.new_event_loop()
        self.event = asyncio.Event()
        self.users = set()

        async def send_loop():
            while self.started:
                await asyncio.sleep(0.008)
                pass

        async def handle_messages(websocket, path):
            @ee.on('update_colors')
            def update_colors():
                rgb = hsv_to_rgb(self.state.hsv)
                message = 'calibration;' + ';'.join(rgb.astype('str'))
                self.sendLoop.create_task(asyncio.wait([user.send(message) for user in self.users]))
            @ee.on('update_camera_draw')
            def update_camera_draw():
                message = 'camera;draw;' + str(int(self.state.drawBall))
                self.sendLoop.create_task(asyncio.wait([user.send(message) for user in self.users]))
            @ee.on('update_camera_follow')
            def update_camera_follow():
                message = 'camera;follow;' + str(int(self.state.followBall))
                self.sendLoop.create_task(asyncio.wait([user.send(message) for user in self.users]))

            self.users.add(websocket)
            try:
                while self.started:
                    msg = await websocket.recv()
                    ee.emit('command_received', msg.split(';'))
            except Exception:
                pass
            finally:
                self.users.remove(websocket)
        self.ws = websockets.serve(handle_messages, '0.0.0.0', 1337, loop=self.receiveLoop)
        self.s = send_loop()

    def start(self):
        if self.started:
            print('[!] Websocket thread has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        self.send_thread = threading.Thread(target=self.update_send, args=())
        self.send_thread.start()
        return self

    def update(self):
        asyncio.set_event_loop(self.receiveLoop)
        self.ws = asyncio.get_event_loop().run_until_complete(self.ws)
        asyncio.get_event_loop().run_forever()
    
    def update_send(self):
        asyncio.set_event_loop(self.sendLoop)
        asyncio.get_event_loop().run_until_complete(self.s)
        asyncio.get_event_loop().run_forever()

    def stop(self):
        self.started = False
        self.thread.join()
        self.send_thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.ws.close()
        asyncio.run_until_complete(self.ws.wait_closed())

class ServoControl:
    @property
    def camera_abs_x_rot(self):
        return self._camera_abs_x_rot
    @camera_abs_x_rot.setter
    def camera_abs_x_rot(self, val):
        self._camera_abs_x_rot = val
        self.pan_servo.write(self._camera_abs_x_rot)

    @property
    def camera_abs_y_rot(self):
        return self._camera_abs_y_rot
    @camera_abs_y_rot.setter
    def camera_abs_y_rot(self, val):
        self._camera_abs_y_rot = val
        self.tilt_servo.write(self._camera_abs_y_rot)

    def __init__(self, state):
        self.bw = back_wheels.Back_Wheels()
        self.fw = front_wheels.Front_Wheels()
        self.pan_servo = Servo.Servo(1)
        self.tilt_servo = Servo.Servo(2)
        sunfounder.setup()

        self.fw.turning_offset = -55
        self.pan_servo.offset = 0
        self.tilt_servo.offset = 0

        self.bw.speed = 0
        self.fw.turn(90)
        self.camera_abs_x_rot = 90
        self.camera_abs_y_rot = 90

        self.state = state
        self.follow_ball = False

    def handle_command(self, cmd):
        if cmd[0] == 'joyR':
            strength = int(cmd[1]) / 100.0
            self.fw.turn(90 + 45 * strength)
        elif cmd[0] == 'joyL':
            angle = int(cmd[1])
            strength = int(cmd[2]) / 100.0
            speed = max(0, min(100, int(100 * strength)))
            self.bw.speed = speed
            if angle == 90: # up
                self.bw.forward()
            elif angle == 270: # down
                self.bw.backward()
        elif cmd[0] == 'joyC': # manual camera
            speed_x = int(cmd[1])
            speed_y = int(cmd[2])
            self.camera_abs_x_rot -= speed_x
            self.camera_abs_y_rot -= speed_y
        elif cmd[0] == 'setC': # set angles
            angle_x = int(cmd[1])
            self.camera_abs_x_rot = angle_x
            self.camera_abs_y_rot = 90

    def update_ball_camera(self): # NOTE: printing in this callback is going to cause serious lag spikes, dunno why
        (x, y, radius) = self.state.ball_props()
        w, h = self.state.resize_resolution()
        dx, dy = x-w/2, y-h/2
        
        abs_dx = math.fabs(dx)
        abs_dy = math.fabs(dy)
        if abs_dx > 30:
            speed_x = np.interp(abs_dx, [30, 256/2], [1, 10])
            self.camera_abs_x_rot -= speed_x * np.sign(dx)
            # self.pan_servo.write(self.camera_abs_x_rot)
        if abs_dy > 30:
            speed_y = np.interp(abs_dy, [30, 192/2], [1, 10])
            self.camera_abs_y_rot -= speed_y * np.sign(dy)
            # self.tilt_servo.write(self.camera_abs_y_rot)
        if self.state.followBall:
            self.fw.turn(180 - self.camera_abs_x_rot)
            if (self.camera_abs_y_rot - 90) > 45:
                return
            if radius > 75:
                self.bw.speed = 25
                self.bw.backward()
            elif 50 > radius >= 25:
                self.bw.speed = 10
                self.bw.forward()
            elif 2 < radius < 25:
                self.bw.speed = 66
                self.bw.forward()
            else:
                self.bw.stop()
        else:
            self.bw.stop()
            self.camera_abs_x_rot = 90
            self.camera_abs_y_rot = 90

class State:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = 640
        self.height = 480
        self.resizeWidth = 256
        self.resizeHeight = 192
        self.calibrate = False
        self.processCalibration = False
        self.hsv = (0, 100, 100)
        self.followBall = False
        self.drawBall = False
        self.stabilise = False

        #ball properties
        self.x = 0
        self.y = 0
        self.radius = 0

    def handle_command(self, cmd):
        if cmd[0] == 'camera':
            if cmd[1] == 'toggle_follow':
                self.followBall = not self.followBall
                ee.emit("update_camera_follow")
            elif cmd[1] == "toggle_draw":
                self.drawBall = not self.drawBall
                ee.emit("update_camera_draw")
            elif cmd[1] == 'calibrate':
                self.calibrate = True
            elif cmd[1] == 'calibrate_done':
                self.processCalibration = True
            elif cmd[1] == 'stabilise_true':
                self.stabilise = True
            elif cmd[1] == 'stabilise_false':
                self.stabilise = False

    def resolution(self):
        return (self.width, self.height)

    def resize_resolution(self):
        return (self.resizeWidth, self.resizeHeight)
    
    def ball_props(self):
        return (self.x, self.y, self.radius)

    def set_ball_props(self, setX, setY, setR):
        self.x, self.y, self.radius = setX, setY, setR

def resize_buffer(buffer, newWidth, newHeight):
    return cv2.resize(buffer, (newWidth, newHeight))

def round_bb(bb):
    return [round(v) for v in bb]

def calc_sized_bb(resW, resH, percent):
    return (resW/4, percent, resW-resW/4, resH-percent)

def calc_dominant_color(buffer):
    # k-means to get dominant color
    pixels = np.float32(buffer.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    # opencv uses BGR, therefore order is rotated here
    # also this needs to be wrapped in 3 arrays for it to work
    col = np.uint8([[[dominant[2].item(), dominant[1].item(), dominant[0].item()]]])
    return cv2.cvtColor(col, cv2.COLOR_RGB2HSV)[0][0][:]

def hsv_to_rgb(hsv):
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0][:]

def scale_circle(circle, oldRes, newRes):
    x, y, radius = circle
    (oldW, oldH) = oldRes
    (newW, newH) = newRes
    return (x/oldW*newW, y/oldH*newH, radius/oldW*newW)

def int_circle(circle):
    x, y, r = circle
    return (int(x), int(y), int(r))

#-----------------------------------------------------------#
#-----------------------------------------------------------#
#-----------------------------------------------------------#

def calibrate(buffer, x, y, w, h):    
    cutout = buffer[y:h, x:w]
    return calc_dominant_color(cutout)

def find_ball(originalFrame, hsv, stabilise=False):
    (h, s, v) = hsv
    lowerBound = np.array([h-10, 100, 100])
    upperBound = np.array([h+10, 255, 255])
    activeFrm = originalFrame
    if stabilise:
        activeFrm = cv2.GaussianBlur(activeFrm, (9, 9), 0)
    hsv = cv2.cvtColor(activeFrm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    if stabilise:
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    masked = cv2.bitwise_and(activeFrm, activeFrm, mask=mask)

    _, cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cont = max(cnts, key=cv2.contourArea) # take the biggest contour
        M = cv2.moments(cont)
        
        (x, y), radius = cv2.minEnclosingCircle(cont) # make a circle out of it
        return True, (x, y, radius)
    else:
        return False, (0, 0, 0)

def jpg_bytes(to_jpg):
    _, jpg = cv2.imencode('.jpg', to_jpg)
    return jpg.tobytes()

vs = VideoCaptureAsync(0)
time.sleep(2.0)

someState = State()
pt = ProcessingThread(vs, someState)
wt = WebsocketThread(someState)
sc = ServoControl(someState)

ee = BaseEventEmitter()
sendee = AsyncIOEventEmitter(loop=wt.sendLoop)
@ee.on('command_received')
def command_received(cmd):
    someState.handle_command(cmd)
    sc.handle_command(cmd)
@ee.on('ball_found')
def ball_found():
    sc.update_ball_camera()

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

def stream():
    global pt

    while True:
        outputFrame = pt.processed_frame()
        if outputFrame is not None:
            time.sleep(1.0/24.0)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes(outputFrame) + b'\r\n')

@app.route("/video_stream")
def video_stream():
	# return the response generated along with the specific media
	# type (mime type)

	return Response(stream(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    vs.start()
    wt.start()
    pt.start()

    app.run(host="0.0.0.0", port=8080, debug=False,
		threaded=True, use_reloader=False)

pt.stop()
wt.stop()
vs.stop()
