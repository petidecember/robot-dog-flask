from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2
import time
import numpy as np
import websockets
import asyncio
from pyee import BaseEventEmitter

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
        self.loop = asyncio.new_event_loop()
        self.event = asyncio.Event()

        async def receive_messages(websocket, path):
            @ee.on('update_colors')
            def update_colors():
                rgb = hsv_to_rgb(self.state.hsv)
                print(rgb)
                self.loop.run_until_complete(websocket.send('calibration;' + ';'.join(rgb.astype('str'))))
            try:
                while self.started:
                    msg = await websocket.recv()
                    if(type(msg) == str):
                        cmd = msg.split(';')
                        if cmd[0] == 'camera':
                            if cmd[1] == 'ball_true':
                                self.state.followBall = True
                            elif cmd[1] == 'ball_false':
                                self.state.followBall = False
                            elif cmd[1] == "draw_true":
                                self.state.drawBall = True
                            elif cmd[1] == "draw_false":
                                self.state.drawBall = False
                            elif cmd[1] == 'calibrate':
                                self.state.calibrate = True
                            elif cmd[1] == 'calibrate_done':
                                self.state.processCalibration = True
                            elif cmd[1] == 'stabilise_true':
                                self.state.stabilise = True
                            elif cmd[1] == 'stabilise_false':
                                self.state.stabilise = False
                        pass

            except websockets.exceptions.ConnectionClosedOK:
                print("Closing websocket connection")
        self.ws = websockets.serve(receive_messages, '0.0.0.0', 1338, loop=self.loop)

    def start(self):
        if self.started:
            print('[!] Websocket thread has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        asyncio.set_event_loop(self.loop)
        self.ws = asyncio.get_event_loop().run_until_complete(self.ws)
        asyncio.get_event_loop().run_forever()

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.ws.close()
        asyncio.run_until_complete(self.ws.wait_closed())

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

    cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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
ee = BaseEventEmitter()

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
            time.sleep(1.0/20.0)
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

    app.run(host="0.0.0.0", port=8080, debug=True,
		threaded=True, use_reloader=False)

pt.stop()
wt.stop()
vs.stop()