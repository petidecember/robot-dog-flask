import cv2
import math
import websockets
import asyncio
import threading
import sunfounder
import numpy as np
from aiohttp import web
from pyee import BaseEventEmitter, AsyncIOEventEmitter
from threading import Thread
from sunfounder import front_wheels, back_wheels
from sunfounder.SunFounder_PCA9685 import Servo

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
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

vs = VideoCaptureAsync(0)
ee2 = AsyncIOEventEmitter(loop=asyncio.get_event_loop()) # async event emitter to send messages
ee = BaseEventEmitter() # event emitter to receive messages

class VideoImageTrack:
    def __init__(self, config):
        super().__init__()
        self.config = config

    async def recv(self):
        if not self.config.active:
            return None
        ret, frm = vs.read()
        frm = self.process_frame(self.config, frm)
        return frm

    def process_frame(self, config, frm):
        resWidth, resHeight = 256, 192
        buffer = cv2.resize(frm, (resWidth, resHeight))
        if config.calibrate:
            a = resHeight*0.2
            initBB = (resWidth/4, a, resWidth-resWidth/4, resHeight-a)
            (x, y, w, h) = [round(v) for v in initBB]
            cv2.rectangle(buffer, (x, y), (w, h), color=(255, 0, 0), thickness=4)
            if config.process_calibration:
                cutout = buffer[y:h, x:w]

                # k-means to get dominant color
                pixels = np.float32(cutout.reshape(-1, 3))

                n_colors = 1
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS

                _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)

                dominant = palette[np.argmax(counts)]
                # opencv uses BGR, therefore order is rotated here
                # also this needs to be wrapped in 3 arrays for it to work
                col = np.uint8([[[dominant[2].item(), dominant[1].item(), dominant[0].item()]]])
                config.hsv = cv2.cvtColor(col, cv2.COLOR_RGB2HSV)[0][0][:]
                config.process_calibration = False
                config.calibrate = False
            col = cv2.cvtColor(np.uint8([[config.hsv]]), cv2.COLOR_HSV2RGB)[0][0][:]
            ee2.emit('send_msg', 'calibration;' + ';'.join(col.astype('str')))
        if config.follow_ball:
            (h, s, v) = config.hsv
            lowerBound = np.array([h-10, 100, 100])
            upperBound = np.array([h+10, 255, 255])
            active_frm = frm
            draw_frm = buffer
            if config.stabilise:
                active_frm = cv2.GaussianBlur(active_frm, (9, 9), 0)
            hsv = cv2.cvtColor(active_frm, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lowerBound, upperBound)
            if config.stabilise:
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
            masked = cv2.bitwise_and(active_frm, active_frm, mask=mask)

            _, cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cont = max(cnts, key=cv2.contourArea) # take the biggest contour
                M = cv2.moments(cont)
                ((x, y), radius) = cv2.minEnclosingCircle(cont) # make a circle out of it
                ((x, y), radius) = ((x/config.width*resWidth, y/config.height*resHeight), radius/config.width*resWidth) # then resize to fit on the buffer
                ee.emit("ball_found", ((x, y), radius))
                if config.draw_ball:
                    cv2.circle(draw_frm, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.circle(draw_frm, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        
        _, jpg = cv2.imencode('.jpg', buffer)
        return jpg.tobytes()


class CameraProcessConfig():
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.active = True
        self.calibrate = False
        self.process_calibration = False
        self.follow_ball = False
        self.draw_ball = False
        self.stabilise = False
        self.hsv = (0, 100, 100)
        
        @ee.on('message')
        def message(msg):
            cmd = msg.split(';')
            if cmd[0] == 'camera':
                if cmd[1] == 'true':
                    self.active = True
                    ee2.emit('send_msg', 'camera;{0};{1}'.format(int(self.width), int(self.height))) # aspect ratio calculation
                elif cmd[1] == 'false':
                    self.active = False
                elif cmd[1] == 'ball_true':
                    self.follow_ball = True
                elif cmd[1] == 'ball_false':
                    self.follow_ball = False
                elif cmd[1] == "draw_true":
                    self.draw_ball = True
                elif cmd[1] == "draw_false":
                    self.draw_ball = False
                elif cmd[1] == 'calibrate':
                    self.calibrate = True
                elif cmd[1] == 'calibrate_done':
                    self.process_calibration = True
                elif cmd[1] == 'stabilise_true':
                    self.stabilise = True
                elif cmd[1] == 'stabilise_false':
                    self.stabilise = False
            pass


class StreamThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.pcs = set()
        self.config = CameraProcessConfig(640, 480)
        self.stream = VideoImageTrack(self.config)
        self.users = set()

    def run(self):
        async def stream(websocket, path):
            try:
                self.users.add(websocket)
                while len(self.users):
                        if not self.config.active:
                            return
                        frame = await self.stream.recv()
                        for user in self.users:    
                            await user.send(frame)
            except websockets.exceptions.ConnectionClosed:
                self.users.remove(websocket)
        
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.ws = websockets.serve(stream, '0.0.0.0', 1338)
        asyncio.get_event_loop().run_until_complete(self.ws)
        asyncio.get_event_loop().run_forever()
        pass


class WebThread(Thread):
    def __init__(self):
        Thread.__init__(self)
    
    def run(self):
        async def index(request):
            content = open("index.html", "r").read()
            return web.Response(content_type="text/html", text=content)

        async def on_shutdown(app):
            coros = [pc.close() for pc in pcs]
            await asyncio.gather(*coros)
            pcs.clear()

        async def start_server():
            app = web.Application()
            app.on_shutdown.append(on_shutdown)
            app.router.add_get('/', index)
            # app.router.add_get('/main.css', stylesheet)
            # app.router.add_get('/client.js', javascript)
            # app.router.add_get('/app', appDownload) # replaced with static
            app.router.add_static('/static/', path='/home/pi/Documents/robot-dog/static', name='static')
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8080)
            await site.start()
        
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(start_server())
        asyncio.get_event_loop().run_forever()


class ControlThread(Thread):
    @property
    def camera_abs_x_rot(self):
        return self._camera_abs_x_rot
    @camera_abs_x_rot.setter
    def camera_abs_x_rot(self, val):
        self._camera_abs_x_rot = val
        # print(val)
        self.pan_servo.write(self._camera_abs_x_rot)

    @property
    def camera_abs_y_rot(self):
        return self._camera_abs_y_rot
    @camera_abs_y_rot.setter
    def camera_abs_y_rot(self, val):
        self._camera_abs_y_rot = val
        self.tilt_servo.write(self._camera_abs_y_rot)

    def __init__(self):
        Thread.__init__(self)
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
        # self.pan_servo.write(self.camera_abs_x_rot)
        # self.tilt_servo.write(self.camera_abs_y_rot)

        self.follow_ball = False

    def run(self):
        @ee.on('message')
        def message(msg):
            cmd = msg.split(';')
            if cmd[0] == 'joyR':
                angle = int(cmd[1])
                strength = int(cmd[2]) / 100.0
                if angle == 0: # left
                    self.fw.turn(89 + 45 * strength)
                elif angle == 180: # right
                    self.fw.turn(89 - 45 * strength)
                else: # neutral
                    self.fw.turn(89+(90*(angle/100)))
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

                # self.pan_servo.write(self.camera_abs_x_rot)
                # self.tilt_servo.write(self.camera_abs_y_rot)
            elif cmd[0] == 'setC': # set angles
                angle_x = int(cmd[1])
                self.camera_abs_x_rot = angle_x
                self.camera_abs_y_rot = 90

                # self.pan_servo.write(self.camera_abs_x_rot)
                # self.tilt_servo.write(self.camera_abs_y_rot)
            elif cmd[0] == 'offset':
                offset = int(cmd[1])
                self.fw.turning_offset = offset
            elif cmd[0] == 'camera':
                if cmd[1] == 'ball_true':
                    self.follow_ball = True
                elif cmd[1] == 'ball_false':
                    self.follow_ball = False
                    self.bw.stop()
                    self.camera_abs_x_rot = 90
                    self.camera_abs_y_rot = 90 # TODO: remove redundance

                    # self.pan_servo.write(self.camera_abs_x_rot)
                    # self.tilt_servo.write(self.camera_abs_y_rot)
        @ee.on('ball_found')
        def ball_found(data): # NOTE: printing in this callback is going to cause serious lag spikes, dunno why
            ((x, y), radius) = data
            dx, dy = x-256/2, y-192/2 # TODO: remove hardcoded buffer sizes
            
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
            if self.follow_ball:
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
            # print(dx, dy)


USERS = set()


async def echo(websocket, path):
    USERS.add(websocket)
    try:
        @ee2.on('send_msg')
        async def send_msg(msg):
            for user in USERS:
                await user.send(msg)
        while True:
            msg = await websocket.recv()
            ee.emit('message', msg)
    except websockets.exceptions.ConnectionClosed:
        USERS.remove(websocket)


def main():
    wt = WebThread()
    st = StreamThread()
    ct = ControlThread()
    ws = websockets.serve(echo, '0.0.0.0', 1337)
    try:
        vs.start()
        wt.setName("Web Server")
        wt.start()

        st.setName("Camera Stream")
        st.start()

        ct.setName("Servo Control")
        ct.start()
        
        ws = asyncio.get_event_loop().run_until_complete(ws)
        asyncio.get_event_loop().run_forever()
        pass
    except KeyboardInterrupt:
        pass
    finally:
        vs.stop()
        wt.join()
        st.join()
        ws.close()
        print('cleanup')
        pass


if __name__ == '__main__':
    main()
