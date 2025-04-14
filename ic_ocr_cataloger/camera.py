import asyncio
import logging
from asyncio import Event
from collections import deque

import cv2
import numpy as np
from PIL import Image

from ic_ocr_cataloger.filters import AlphaBetaMod

phasher = cv2.img_hash.PHash.create()


class Camera:

    def __init__(self, app, config):
        self.cam = None
        self.config = config
        self.frame_height = None
        self.frame_width = None
        self.app = app
        self.cur_cv2_image: None | np.ndarray = None
        self.cur_image: None | Image.Image = None
        self.cur_phash = None
        self.hash_diff = 0
        self.frame_ready = Event()
        self.prev_phash = None
        self.filters = deque(maxlen=5)
        self.alphabeta = AlphaBetaMod()

    def open_camera(self):
        self.cam = cam = cv2.VideoCapture(self.config["main"].getint("cam_device", 0))
        logging.info(
            "Camera opened: %dx%d %d fps: %s",
            cam.get(cv2.CAP_PROP_FRAME_WIDTH),
            cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cam.get(cv2.CAP_PROP_FPS),
            cam.get(cv2.CAP_PROP_HW_DEVICE),
        )
        cam.set(cv2.CAP_PROP_FPS, self.config["main"].getint("cam_fps", 30))
        x, y = self.config["main"].get("cam_resolution", "1920x1080").split("x")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
        self.frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(
            "Camera configured: %dx%d %d fps: %s",
            cam.get(cv2.CAP_PROP_FRAME_WIDTH),
            cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cam.get(cv2.CAP_PROP_FPS),
            cam.get(cv2.CAP_PROP_HW_DEVICE),
        )

    async def stream_frames(self):
        while not asyncio.current_task().cancelled():
            rc, frame = await asyncio.to_thread(self.cam.read)
            if not rc:
                continue
            yield frame

    async def live_video(self):
        self.open_camera()
        async for frame in self.stream_frames():
            self.app.counter.update(frame=1)

            # Check for frame similarity before applying filters
            self.compute_similarity(frame)
            if self.hash_diff > 10:
                self.app.reset()

            if self.filters:
                for img_filter in self.filters:
                    try:
                        frame = img_filter(frame)
                    except Exception as e:
                        logging.exception(e)
                        self.filters.remove(img_filter)
                        logging.warning(f"Removed filter: {img_filter}")

            self.cur_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.cur_cv2_image = frame
            self.frame_ready.set()

    def reset(self):
        self.frame_ready.clear()
        self.cur_cv2_image = None
        self.cur_image = None
        self.cur_phash = None
        self.hash_diff = 0
        self.prev_phash = None
        self.alphabeta.reset()
        self.filters.clear()

    def compute_similarity(self, frame):
        self.cur_phash = phasher.compute(
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                dsize=None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_CUBIC,
            )
        )
        prev = self.prev_phash
        self.prev_phash = self.cur_phash
        if prev is None:
            self.hash_diff = 0
            return
        self.hash_diff = phasher.compare(prev, self.cur_phash)

    def invert(self):
        self.filters.append(lambda i: cv2.bitwise_not(i))

    def grayscale(self):
        self.filters.append(
            lambda i: cv2.cvtColor(
                cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )
        )

    def blur(self):
        self.filters.append(lambda i: cv2.GaussianBlur(i, (5, 5), 0))

    def sharpen(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.filters.append(lambda i: cv2.filter2D(i, -1, kernel))

    def contrast(self, value):
        if self.alphabeta not in self.filters:
            self.filters.append(self.alphabeta)
        # alpha 1  beta 0      --> no change
        # 0 < alpha < 1        --> lower contrast
        # alpha > 1            --> higher contrast
        # -127 < beta < +127   --> good range for brightness values
        self.alphabeta.a = max(0.1, self.alphabeta.a + (value * 0.05))

    def brightness(self, value):
        if self.alphabeta not in self.filters:
            self.filters.append(self.alphabeta)
        self.alphabeta.b = min(127, max(-127, self.alphabeta.b + (value * 1)))

    def release(self):
        self.cam.release()
