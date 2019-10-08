from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web
from av import VideoFrame
import cv2
import asyncio
import json
import re

class VideoImageTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, img = vs.read()
        img = cv2.resize(img, (256, 192))
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        return frame

async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)


async def stylesheet(request):
    content = open("main.css", "r").read()
    return web.Response(content_type="text/css", text=content)


async def javascript(request):
    content = open("client.js", "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    print(params["sdp"])
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        pc.addTrack(VideoImageTrack())

    answer = await pc.createAnswer()
    
    await pc.setLocalDescription(answer)
    
    # print(answer.sdp)
    # sdp = ''.join(answer.sdp).split('\n')
    # for i in range(len(sdp)):
    #     line = sdp[i]
    #     if line.startswith("m="):
    #         words = line.split(' ')
    #         length = len(words) - 1
    #         words[length] = words[length].replace('\r', '')
    #         words[length], words[length-1] = words[length-1], words[length]
    #         line = ' '.join(words)
    # print(sdp, sep='\n')
    # answer.sdp = '\n'.join(sdp)
    
    #print('\n'.join(sdp))

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

pcs = set()

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

vs = cv2.VideoCapture(0)

if __name__ == '__main__':
    if not vs.isOpened():
        print('Could not open webcam')
        exit()
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/main.css', stylesheet)
    app.router.add_get('/client.js', javascript)
    app.router.add_post(path='/offer', handler=offer)
    web.run_app(app, port=8080, ssl_context=None)