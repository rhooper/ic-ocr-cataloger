import asyncio
import time
from datetime import datetime
from typing import NamedTuple

import cv2
from PIL import ImageFont

from ic_ocr_cataloger.app import BIG_FONT


async def async_read_key(timeout: float = 1):
    end_time = time.time() + timeout
    while time.time() < end_time:
        key = cv2.waitKey(1)
        if key > -1:
            return key
        await asyncio.sleep(0.01)
    return -1


class FlashMessage(NamedTuple):
    expire: datetime
    text: str
    color: tuple[int, int, int] = (128, 128, 128)
    font: ImageFont = BIG_FONT
