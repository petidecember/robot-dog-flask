#from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2
import time
import numpy as np
import websockets
import signal

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
        while True:
            resWidth, resHeight = someState.resizeWidth, someState.resizeHeight
            grabbed, frame = vs.read()
            
            resizedFrame = resize(frame, resWidth, resHeight)
            
            if someState.calibrate:
                a = resHeight*0.2
                calibBB = calc_sized_bb(resWidth, resHeight, a)
                (x, y, w, h) = round_bb(calibBB)
                cv2.rectangle(resizedFrame, (x, y), (w, h), color=(255, 0, 0), thickness=4)
                someState.hsv = calibrate(resizedFrame, someState, x, y, w, h)
            if someState.followBall:
                (someState.x, someState.y), someState.radius = find_ball(frame, resizedFrame, someState, resWidth, resHeight)
            if someState.drawBall:
                cv2.circle(resizedFrame, (someState.x, someState.y), 5, (0, 0, 255), -1)
                cv2.circle(resizedFrame, (someState.x, someState.y), someState.radius, (0, 255, 255), 2)

            self.processedFrame = resizedFrame

    def processed_frame(self):
        return self.processedFrame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        pass

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
        self.followBall = True
        self.drawBall = True
        self.stabilise = True

        #ball properties
        self.x = 0
        self.y = 0
        self.radius = 0

def round_bb(bb):
    return [round(v) for v in bb]

def resize(buffer, newres, newheight):
    return cv2.resize(buffer, (newres, newheight))

def calc_sized_bb(resw, resh, percent):
    return (resw/4, percent, resw-resw/4, resh-percent)

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

#-----------------------------------------------------------#
#-----------------------------------------------------------#
#-----------------------------------------------------------#

def calibrate(buffer, state, x, y, w, h):
    if state.processCalibration:
        state.processCalibration = False
        state.calibrate = False
        cutout = buffer[y:h, x:w]
        return calc_dominant_color(cutout)

def temp():
    col = cv2.cvtColor(np.uint8([[config.hsv]]), cv2.COLOR_HSV2RGB)[0][0][:] # TODO Setup controls and this

def find_ball(originalFrame, resizedFrame, state, resWidth, resHeight):
    (h, s, v) = state.hsv
    lowerBound = np.array([h-10, 100, 100])
    upperBound = np.array([h+10, 255, 255])
    activeFrm = originalFrame
    drawFrm = resizedFrame
    if state.stabilise:
        activeFrm = cv2.GaussianBlur(activeFrm, (9, 9), 0)
    hsv = cv2.cvtColor(activeFrm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    if state.stabilise:
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    masked = cv2.bitwise_and(activeFrm, activeFrm, mask=mask)

    _, cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cont = max(cnts, key=cv2.contourArea) # take the biggest contour
        M = cv2.moments(cont)
        
        ((x, y), radius) = cv2.minEnclosingCircle(cont) # make a circle out of it
        ((x, y), radius) = ((x/state.width*resWidth, y/state.height*resHeight), radius/state.width*resWidth) # then resize to fit on the buffer
        return ((int(x), int(y)), int(radius))
    else:
        return((state.x, state.y), state.radius)

def jpg_bytes(to_jpg):
    _, jpg = cv2.imencode('.jpg', to_jpg)
    return jpg.tobytes()

vs = VideoCaptureAsync(0)
time.sleep(2.0)

someState = State()
pt = ProcessingThread(vs, someState)

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
    pt.start()

    app.run(host="0.0.0.0", port=8080, debug=True,
		threaded=True, use_reloader=False)

pt.stop()
vs.stop()