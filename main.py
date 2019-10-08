#from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2
import time
import numpy as np

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

class State:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrate = False
        self.process_calibration = False
        self.hsv = (0, 100, 100)
        self.follow_ball = False
        self.stabilise = False

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

def calibrate(buffer, state, resWidth, resHeight):
    a = resHeight*0.2
    calibBB = calc_sized_bb(resWidth, resHeight, a)
    (x, y, w, h) = round_bb(calibBB)
    cv2.rectangle(buffer, (x, y), (w, h), color=(255, 0, 0), thickness=4)
    if state.process_calibration:
        state.process_calibration = False
        state.calibrate = False
        cutout = buffer[y:h, x:w]
        return calc_dominant_color(cutout)

def temp():
    col = cv2.cvtColor(np.uint8([[config.hsv]]), cv2.COLOR_HSV2RGB)[0][0][:] # TODO Setup controls and this

def find_ball(original_frame, resized_frame, state, resWidth, resHeight):
    (h, s, v) = state.hsv
    lowerBound = np.array([h-10, 100, 100])
    upperBound = np.array([h+10, 255, 255])
    active_frm = original_frame
    draw_frm = resized_frame
    if state.stabilise:
        active_frm = cv2.GaussianBlur(active_frm, (9, 9), 0)
    hsv = cv2.cvtColor(active_frm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    if state.stabilise:
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    masked = cv2.bitwise_and(active_frm, active_frm, mask=mask)

    _, cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cont = max(cnts, key=cv2.contourArea) # take the biggest contour
        M = cv2.moments(cont)
        ((x, y), radius) = cv2.minEnclosingCircle(cont) # make a circle out of it
        ((x, y), radius) = ((x/state.width*resWidth, y/state.height*resHeight), radius/state.width*resWidth) # then resize to fit on the buffer
        #ee.emit("ball_found", ((x, y), radius))
        return ((x, y), radius)
        if state.draw_ball:
            cv2.circle(draw_frm, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.circle(draw_frm, (int(x), int(y)), int(radius), (0, 255, 255), 2)

def jpg_bytes(to_jpg):
    _, jpg = cv2.imencode('.jpg', to_jpg)
    return jpg.tobytes()

vs = VideoCaptureAsync(0)
time.sleep(2.0)

outputFrame = None
someState = State()

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

def t():
    resWidth, resHeight = 256, 192
    resizedFrame = resize(outputFrame, resWidth, resHeight)
    if someState.calibrate:
        someState.hsv = calibrate(resizedFrame, someState, resWidth, resHeight)
    if someState.follow_ball:
        (someState.x, someState.y), someState.radius = find_ball(outputFrame, resizedFrame, someState, resWidth, resHeight)

def stream():
    global outputFrame, vs

    while True:
        time.sleep(1.0/20.0)
        grabbed, outputFrame = vs.read()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes(outputFrame) + b'\r\n')

@app.route("/video_stream")
def video_stream():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(stream(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    vs.start()

    app.run(host="0.0.0.0", port=8080, debug=True,
		threaded=True, use_reloader=False)

vs.stop()