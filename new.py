import cv2
import websockets
import asyncio
import threading
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web
from pyee import BaseEventEmitter, AsyncIOEventEmitter
from av import VideoFrame
from threading import Thread
import numpy as np


class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
ee = AsyncIOEventEmitter(loop=asyncio.get_event_loop())


class VideoImageTrack(VideoStreamTrack):
    def __init__(self, config):
        super().__init__()
        self.config = config

    async def recv(self):
        if not self.config.active:
            return None
        pts, time_base = await self.next_timestamp()
        ret, frm = vs.read()
        frm = self.process_frame(self.config, frm)
        frame = VideoFrame.from_ndarray(frm, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def recvTcp(self):
        if not self.config.active:
            return None
        ret, frm = vs.read()
        frm = self.process_frame(self.config, frm)
        return frame

    def process_frame(self, config, frm):
        buffer = cv2.resize(frm, (256, 192))
        if config.calibrate:
            a = 192*0.2
            initBB = (256/4, a, 256-256/4, 192-a)
            (x, y, w, h) = [round(v) for v in initBB]
            cv2.rectangle(buffer, (x, y), (w, h), color=(255, 0, 0), thickness=4)
            if config.process_calibration:
                cutout = frame[y:h, x:w]

                pixels = np.float32(cutout.reshape(-1, 3))

                n_colors = 1
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS

                _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)

                dominant = palette[np.argmax(counts)]
                col = np.uint8([[[dominant[0].item(), dominant[1].item(), dominant[2].item()]]])
                col = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
                config.hsv = col[0][0][:]
                config.process_calibration = False
                config.calibrate = False
        if config.follow_ball:
            (h, s, v) = config.hsv
            lower = np.array([h-10, s, 100])
            upper = np.array([h+10, 255, 255])
            active_frm = buffer

            blurred = cv2.GaussianBlur(active_frm, (11, 11), 0)
            hsv = cv2.cvtColor(active_frm, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            masked = cv2.bitwise_and(active_frm, active_frm, mask=mask)

            _, cnts, hierarchy = cv2.findContours(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cont = max(cnts, key=cv2.contourArea)
                cv2.drawContours(frame, [cont], -1, (0, 0, 255), 3)
                M = cv2.moments(cont)
                ((x, y), radius) = cv2.minEnclosingCircle(cont)
                cv2.circle(active_frm, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.circle(active_frm, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        
        #_, jpg = cv2.imencode('.jpg', buffer)
        # ee.emit("frame_done", buffer)
        return buffer


class CameraProcessConfig():
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.active = True
        self.calibrate = False
        self.process_calibration = False
        self.follow_ball = False
        self.hsv = (0, 100, 100)
        
        @ee.on('message')
        def message(msg):
            cmd = msg.split(';')
            if cmd[0] == 'camera':
                if cmd[1] == 'true':
                    self.active = True
                    ee.emit('send_msg', 'camera;{0};{1}'.format(int(self.width), int(self.height)))
                elif cmd[1] == 'false':
                    self.active = False
                elif cmd[1] == 'ball_true':
                    self.follow_ball = True
                elif cmd[1] == 'ball_false':
                    self.follow_ball = False
                elif cmd[1] == 'calibrate':
                    self.calibrate = True
                elif cmd[1] == 'calibrate_done':
                    self.process_calibration = True
            pass


class StreamThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.pcs = set()
        self.config = CameraProcessConfig(640, 480)
        self.stream = VideoImageTrack(self.config)
        self.protocol = "Websocket"

    def set_protocol(self, prot, user):
        if self.protocol == prot:
            return

        if prot == "Websocket":

            pass
        elif prot == "WebRTC":
            pass
        self.protocol = prot
        

    def run(self):
        async def offerUdp(sdp, type):
            self.protocol = "WebRTC"
            offer = RTCSessionDescription(sdp=sdp, type=type)
            pc = RTCPeerConnection()
            self.pcs.add(pc)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                print("ICE connection state is %s" % pc.iceConnectionState)
                if pc.iceConnectionState == "failed":
                    await pc.close()
                    self.pcs.discard(pc)

            await pc.setRemoteDescription(offer)
            for t in pc.getTransceivers():
                pc.addTrack(self.stream)

            answer = await pc.createAnswer()
    
            await pc.setLocalDescription(answer)

            ee.emit('send_msg', "{0};{1}".format(pc.localDescription.type, pc.localDescription.sdp))

        async def offerTcp():
            self.protocol = "Websocket"
            

        @ee.on('message')
        def message(msg):
            cmd = msg.split(';')
            if cmd[0] == 'offer':
                mtype = cmd[0]
                sdp = msg.replace('offer;', '')
                asyncio.get_event_loop().create_task(offerUdp(sdp, mtype))
            elif cmd[0] == 'tcp':

            pass


class WebThread(Thread):
    def __init__(self):
        Thread.__init__(self)
    
    def run(self):
        async def index(request):
            content = open("index.html", "r").read()
            return web.Response(content_type="text/html", text=content)

        async def stylesheet(request):
            content = open("main.css", "r").read()
            return web.Response(content_type="text/css", text=content)

        async def javascript(request):
            content = open("client.js", "r").read()
            return web.Response(content_type="application/javascript", text=content)

        async def on_shutdown(app):
            coros = [pc.close() for pc in pcs]
            await asyncio.gather(*coros)
            pcs.clear()

        async def start_server():
            app = web.Application()
            app.on_shutdown.append(on_shutdown)
            app.router.add_get('/', index)
            app.router.add_get('/main.css', stylesheet)
            app.router.add_get('/client.js', javascript)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8080)
            await site.start()
        
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(start_server())
        asyncio.get_event_loop().run_forever()


USERS = set()


async def echo(websocket, path):
    USERS.add(websocket)
    @ee.on('send_msg')
    async def send_msg(msg):
        for user in USERS:
            await user.send(msg)
    while True:
        msg = await websocket.recv()
        ee.emit('message', msg)


def main():
    wt = WebThread()
    st = StreamThread()
    ws = websockets.serve(echo, '0.0.0.0', 1337)
    try:
        vs.start()
        wt.setName("Web Server")
        wt.start()

        st.setName("Camera Stream")
        st.start()
        
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
        vs.release()
        print('cleanup')
        pass


if __name__ == '__main__':
    main()