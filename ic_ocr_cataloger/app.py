import asyncio
import itertools
import json
import logging
import multiprocessing
import re
import textwrap
import time
from asyncio import Event, TimerHandle
from collections import Counter, deque, defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFont
from PIL import ImageDraw
from more_itertools import unique_everseen
from ocrmac import ocrmac

from catalog import FoundPart, Catalog, PartInfo
from util import BoundingBox, FlashMessage, count_matches, aggregate

BIG_FONT = ImageFont.truetype("NotoSansNerdFont-Regular", 40)
MAIN_FONT = ImageFont.truetype("Arial", 30)
SMALL_FONT = ImageFont.truetype("Arial", 14)
MEDIUM_FONT = ImageFont.truetype("NotoSansNerdFont-Thin", 18)
FIXED_FONT = ImageFont.truetype("3270NerdFont-Regular", 20)
MATCH_FONT = ImageFont.truetype("NotoSansMNerdFontMono-Light", 25)
RAW_TEXT_FONT = ImageFont.truetype("NotoSansNerdFont-Light", 16)
HUGE_FONT = ImageFont.truetype("NotoSansNerdFont-Regular", 80)
NUM_SEARCH_RESULTS = 16

SENTINEL = object()

phasher = cv2.img_hash.PHash.create()


class AlphaBetaMod:
    """
    Class to adjust brightness and contrast of the image.
    """

    def __init__(self):
        self.a = 1
        self.b = 0
        self.reset()

    def __repr__(self):
        return f"a={self.a:.1f} b={self.b:.1f}"

    def reset(self):
        self.a = 1
        self.b = 0

    def __call__(self, im):
        return cv2.convertScaleAbs(im, alpha=self.a, beta=self.b)


class OCRWorker(multiprocessing.Process):
    """
    Worker process to run (multiple) OCR workers.
    """

    def __init__(self, n, req_queue, res_queue):
        super().__init__()
        self.n = n
        self.req_queue = req_queue
        self.res_queue = res_queue
        self.daemon = True
        self.name = f"ocr_worker_{n}"

    def run(self):
        logging.info("%s", f"OCR Worker no {self.n} ready")
        while True:
            try:
                # if self.req_queue.empty():
                #     pass
                # logging.debug("Waiting for request")
                frame = self.req_queue.get()
                im = Image.fromarray(frame, "RGB")
                if im is None:
                    logging.debug("No image")
                    continue
                ocr = ocrmac.OCR(
                    im,
                    language_preference=["en-US"],
                    framework="vision",
                    recognition_level="accurate",
                )
                result = ocr.recognize()
                if self.res_queue.full():
                    logging.info("Waiting for result queue")
                self.res_queue.put(result)
            except Exception as e:
                logging.exception(e)


async def async_read_key(timeout: float = 1):
    end_time = time.time() + timeout
    while time.time() < end_time:
        key = cv2.waitKey(1)
        if key > -1:
            return key
        await asyncio.sleep(0.01)
    return -1


def format_part_info(
    n_occ: int | str,
    part_string: str,
    part: PartInfo,
    wrap_desc_col: int = 55,
    columns: tuple = ("part_no", "n_occ", "pins", "part_string", "description"),
) -> str:
    description = textwrap.wrap(part.description, wrap_desc_col)
    lines = []
    strings = []
    for field in columns:
        n_cols = sum(len(i) + 3 for i in strings) - 3
        match field:
            case "part_no":
                strings += [f"{part.part_no:<14s}"]
            case "n_occ":
                if isinstance(n_occ, str):
                    strings += [f"{n_occ:<2s}"]
                else:
                    strings += [f"{n_occ:>2d}"]
            case "pins":
                if part.pins in (None, -1, 0, "None"):
                    strings += ["      "]
                else:
                    strings += [f"{str(part.pins):>6s}"]
            case "part_string":
                strings += [f"{part_string:>14s}" if part_string is not None else ""]
            case "description":
                for n, line in enumerate(description):
                    field_val = (" " * n_cols + " | " if n > 0 else "") + line
                    if len(field_val) < wrap_desc_col:
                        field_val += " " * (wrap_desc_col - len(field_val))
                    if n > 0:
                        lines += [field_val]
                    else:
                        strings += [field_val]
            case "flags":
                flags = set(part.flags or [])
                flags.discard("")
                flags.discard("None")
                flags.discard(None)  # noqa
                flags.discard("null")
                strings += [",".join([fl.strip() for fl in flags])]
    return "\n".join([(" | ".join(strings)).strip()] + lines)


class Search:
    SEARCH_ACTION_MODE = True
    NOT_SEARCH_ACTION_MODE = False


class App:
    """
    Main application class for the OCR cataloger.

    Keys:
    - ESC: Reset
    - ENTER: Save part
    - q: Quit
    - r: Reimport catalog
    - g: Grayscale
    - i: Invert
    - l: Gaussian Blur
    - s: Sharpen
    - b: Brightness up
    - B: Brightness down
    - c: Contrast up
    - C: Contrast down
    - x: Reset filters
    - !: Force lock
    - >: Save unknown
    - `: Reset OCR
    - /: Search
        - most keys - add character
        - ESC: Cancel search
        - [ or ]: Add % to search
        - :  ...
        - ': ...
        - 0-9: Select search result
        - UP/DOWN: Select search result
        - LEFT/RIGHT: Page up/down
        When selected:
            - ENTER: Save part
            - ESC exit selected mode
    - v: Toggle verbose
    - .: Sticky
    - -: Pop filter

    """

    def __init__(self, args, config, ocr_req_queue, ocr_res_queue):
        self.later_path = Path(
            config["main"].get("try_later_dir", "ic_try_later").expanduser()
        )
        self.later_path.mkdir(exist_ok=True, parents=True)
        self.opath = Path(
            config["main"].get("saved_images_dir", "ic_saved").expanduser()
        )
        self.config = config
        self.args = args
        self.previous_save = None
        self.current_search_pick = 0
        self._search_task = None
        self.search_results = []
        self.search_result_page = 0
        self.key_buffer = "%"
        self._force_lock = None
        self.best_match: dict[FoundPart, int] | None = None
        self.bounding_boxes: list[BoundingBox] = []
        self.catalog = Catalog()
        self.counter = Counter(n=self.catalog.get_max(), frame=0)
        self.detecting_state = "Searching for parts"
        self.hash_diff = 0
        self.filters = deque(maxlen=5)
        self.flash_messages: deque[FlashMessage] = deque(maxlen=3)
        self.found_part_event = Event()
        self.frame_ready = Event()
        self.cur_image: None | Image.Image = None
        self.cur_cv2_image: None | np.ndarray = None
        self.cur_phash = None
        self.locked_bbox = None
        self.locked_parts: dict[FoundPart, int] = None
        self.ocr_stable = Event()
        self.prev_phash = None
        self.raw_text: str = ""
        self.recently_found: deque[dict[FoundPart, int]] = deque(maxlen=25)
        self.verbose = False
        self.ocr_res_queue: multiprocessing.Queue = ocr_res_queue
        self.ocr_req_queue: multiprocessing.Queue = ocr_req_queue
        self.alphabeta = AlphaBetaMod()
        self.features: dict[str, TimerHandle | None] = {}

        self.cam = cam = cv2.VideoCapture(int(config["main"].get("cam_device", 0)))
        logging.info(
            "Camera opened: %dx%d %d fps: %s",
            cam.get(cv2.CAP_PROP_FRAME_WIDTH),
            cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cam.get(cv2.CAP_PROP_FPS),
            cam.get(cv2.CAP_PROP_HW_DEVICE),
        )
        cam.set(
            cv2.CAP_PROP_FPS,
        )
        x, y = config["main"].get("cam_resolution", "1920x1080").split("x")
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

        self.ocr_pool = [
            OCRWorker(n, ocr_req_queue, ocr_res_queue)
            for n in range(config["main"].getint("ocr_workers", 1))
        ]
        for worker in self.ocr_pool:
            worker.start()

    async def add_feature(
        self, flag: str, timeout: int = 15, timeout_callback: Callable | None = None
    ) -> None:
        # if "flag" in self.features:
        #     self.remove_feature(flag)
        if timeout > 0:
            self.features[flag] = asyncio.get_event_loop().call_later(
                timeout,
                lambda: [
                    self.remove_feature(flag),
                    timeout_callback and timeout_callback(flag),
                ],
            )
        else:
            self.features[flag] = None

    def remove_feature(self, flag: str):
        timer: TimerHandle | None = self.features.get(flag, SENTINEL)
        if timer is SENTINEL:
            return
        if isinstance(timer, TimerHandle):
            timer.cancel()
        del self.features[flag]

    def add_flash_message(
        self, message, delay=15, color=(255, 255, 255), font=BIG_FONT
    ):
        if message not in self.flash_messages:
            self.flash_messages.append(
                FlashMessage(
                    datetime.now() + timedelta(seconds=delay), message, color, font=font
                )
            )

    async def stream_frames(self):
        while not asyncio.current_task().cancelled():
            rc, frame = await asyncio.to_thread(self.cam.read)
            if not rc:
                continue
            yield frame

    async def live_video(self):
        async for frame in self.stream_frames():
            self.counter.update(frame=1)
            if self.filters:
                for img_filter in self.filters:
                    frame = img_filter(frame)
            self.cur_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.cur_cv2_image = frame
            self.frame_ready.set()

    async def ui_task(self):

        cv2.namedWindow("UI", cv2.WINDOW_NORMAL)
        ui = np.zeros((1080, 1920, 3), dtype=np.uint8)

        cv2.putText(
            ui,
            "Waiting for Camera             Q to quit",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
        )

        cv2.imshow("UI", ui)

        while not asyncio.current_task().cancelled():
            await asyncio.sleep(0)

            if self.frame_ready.is_set():
                ui = self.draw_ui(self.cur_image)
                ui_im = cv2.cvtColor(np.array(ui), cv2.COLOR_RGB2BGR)
                cv2.imshow("UI", ui_im)

            key = cv2.waitKey(1)
            if "Search" in self.features:
                await self.typing_search(key)
                continue

            if key == -1:
                await asyncio.sleep(0.01)
                continue

            if key in b"xX" + bytes([27]):
                self.reset()
                self.add_flash_message("Reset", delay=1, color=(255, 0, 0))

            if key in b"qQ":
                break

            if key in b"rR":
                await self.do_reimport()

            if key in (3, 13, 10):
                await self.handle_esc(ui)

            if key in b"gG":
                self.add_flash_message("Grayscale", 2, font=SMALL_FONT)
                self.filters.append(
                    lambda i: cv2.cvtColor(
                        cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
                    )
                )
            if key in b"iI":
                self.add_flash_message("Invert", 2, font=SMALL_FONT)
                self.filters.append(lambda i: cv2.bitwise_not(i))
            if key in b"lL":
                self.add_flash_message("Gaussian Blur", 2, font=SMALL_FONT)
                self.filters.append(lambda i: cv2.GaussianBlur(i, (5, 5), 0))
            if key in b"Ss":
                self.add_flash_message("Sharpen", 2, font=SMALL_FONT)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                self.filters.append(lambda i: cv2.filter2D(i, -1, kernel))
            if key in b"bBcC":
                # Decrease brightness
                if self.alphabeta not in self.filters:
                    self.filters.append(self.alphabeta)
                mod = +1 if chr(key).islower() else -1
                element = "b" if key in b"bB" else "a"
                if element == "a":
                    mod *= 0.05
                    self.alphabeta.a += mod
                else:
                    mod *= 1
                    self.alphabeta.b += mod

                self.add_flash_message(
                    f"adjust {chr(key)}{'-' if mod < 0 else '+'} = {self.alphabeta}",
                    2,
                    font=MEDIUM_FONT,
                )
            if key in b" -" and self.filters:
                self.add_flash_message("Pop", 2, font=SMALL_FONT)
                self.filters.pop()

            if key in b".":
                if "sticky" in self.features:
                    self.remove_feature("sticky")
                else:
                    await self.add_feature("sticky")

            if key in b"!":
                if self._force_lock and not self._force_lock.done():
                    self.add_flash_message(
                        "Force-Lock already pending",
                        5,
                        color=(250, 32, 0),
                        font=MEDIUM_FONT,
                    )
                else:
                    self._force_lock = asyncio.create_task(self.force_lock())

            if key in b">":
                self.add_flash_message("For Later", 4, color=(255, 255, 255))
                self.save_unknown()

            if key in b"`":
                self.ocr_stable.clear()

            if key in b"/":
                self.key_buffer = "%"
                await self.add_feature("Search", timeout=60)

            if key in b"v":
                self.verbose = not self.verbose

    async def handle_esc(self, ui):
        if self.locked_parts:
            self.save_part(
                self.locked_parts,
                ui,
            )
        else:
            self.add_flash_message(
                "No parts locked (! to force-lock)",
                delay=1,
                color=(0, 255, 255),
            )

    async def do_reimport(self):
        stats = self.catalog.reimport()
        logging.info(stats)
        self.add_flash_message(
            "Catalog loaded\n"
            f"{stats["added"]} added, {stats["updated"]} updated, {stats["ignored"]} ignored\n"
            f"{stats["line"]} => {stats["expanded"]}\n"
            f"{stats["files"]} files",
            delay=5,
            color=(255, 255, 0),
            font=FIXED_FONT,
        )
        # self.add_flash_message("Catalog Reloaded", delay=5, color=(255, 255, 0))

    async def typing_search(self, key: int = None):
        await self.add_feature("Search", timeout=60)

        match key, "Action" in self.features:
            case -1, _:  # No key pressed
                return
            case 27, _:  # ESC
                self.remove_feature("Search")
                self.remove_feature("No Boxes")
                self.remove_feature("Action")
                self.key_buffer = "%"
            case 47, Search.NOT_SEARCH_ACTION_MODE:
                await self.add_feature("Action", timeout=60)
            case 0 | 1 | 2 | 3, _:  # Down Up Left Right
                await self.typing_search_subkey(key)
            case 13, Search.SEARCH_ACTION_MODE:  # ENTER
                await self.typing_search_subkey(key)
            case 8 | 127, _:  # BKSP, DEL
                self.key_buffer = self.key_buffer[:-1]
            case 91, _:  # [
                if self.key_buffer.startswith("%"):
                    self.key_buffer = self.key_buffer[1:]
                else:
                    self.key_buffer = "%" + self.key_buffer.lstrip("%")
            case 93, _:  # [
                self.key_buffer = (
                    self.key_buffer[:-1]
                    if self.key_buffer.endswith("%")
                    and not self.key_buffer.startswith("%")
                    else self.key_buffer + "%"
                )
            case 59, _:  # ;
                self.key_buffer = re.sub(
                    r"^(.*(\d{2}))%*(.*)$", r"%\1%\2", self.key_buffer
                )
            case 39, _:  # '
                self.key_buffer = re.sub(r"^[A-Z%]*", "%", self.key_buffer)
            case _, Search.SEARCH_ACTION_MODE:
                await self.typing_search_subkey(key)
            case _ if key != 127 and key > 32:
                self.key_buffer += chr(key).upper()
                if self.key_buffer[-2:].startswith(
                    "%"
                ) and not self.key_buffer.startswith("%"):
                    self.key_buffer = self.key_buffer[:-2] + self.key_buffer[-1] + "%"

        self.search_results = []
        if len(self.key_buffer) > 2:
            if not (
                results := list(
                    self.catalog.search(
                        self.key_buffer,
                        limit=NUM_SEARCH_RESULTS,
                        offset=self.search_result_page * NUM_SEARCH_RESULTS,
                    )
                )
            ):
                if self.search_result_page > 0:
                    self.search_result_page = 0
                    results = list(
                        self.catalog.search(
                            self.key_buffer,
                            limit=NUM_SEARCH_RESULTS,
                            offset=self.search_result_page * NUM_SEARCH_RESULTS,
                        )
                    )

            if len(self.search_results) == 1:
                self.current_search_pick = 1

            for n, (_, r) in enumerate(results):
                if n == self.current_search_pick - 1:
                    color = (255, 255, 0)
                else:
                    color = (255, 255, 255)
                self.search_results.append(
                    [
                        format_part_info(
                            f"{n+1:X}",
                            part_string="",
                            part=r,
                            wrap_desc_col=80,
                            columns=(
                                "n_occ",
                                "part_no",
                                "pins",
                                "description",
                                "flags",
                            ),
                        ),
                        r,
                        color,
                    ]
                )

    def clear_search(self):
        self.remove_feature("Search")
        self.remove_feature("No Boxes")
        self.remove_feature("Action")
        self.key_buffer = "%"
        self.search_results = []
        self.search_result_page = 0
        self.current_search_pick = 0
        self.add_flash_message(
            "Search cleared", delay=1, color=(128, 128, 128), font=SMALL_FONT
        )

    async def typing_search_subkey(self, key=None):
        if not self.search_results:
            return
        if "Action" not in self.features:
            self.current_search_pick = 0
        await self.add_feature("Action", timeout=60)
        await self.add_feature("No Boxes", timeout=15)

        match key:
            case 0 | 1:  # Up
                if key == 0:
                    self.current_search_pick -= 1
                else:
                    self.current_search_pick += 1
                if self.current_search_pick < 1:
                    self.current_search_pick = len(self.search_results)
                elif self.current_search_pick > len(self.search_results):
                    self.current_search_pick = 1
            case 2 | 3:  # Left/Right
                pg = self.search_result_page
                if (
                    len(self.search_results) < NUM_SEARCH_RESULTS
                    and self.search_result_page == 0
                ):
                    return

                if key == 2:
                    self.search_result_page = min(0, self.search_result_page - 1)
                else:
                    self.search_result_page += 1
                if self.search_result_page < 0:
                    self.search_result_page = 0
                if pg != self.search_result_page:
                    self.current_search_pick = 0
            case 27:
                return
            case (
                49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 48 | 97,
                98,
                99,
                100,
                101,
                102,
            ):
                self.current_search_pick = int(chr(key), 16)
            case 13 | 10 if self.current_search_pick:
                pick = self.search_results[self.current_search_pick - 1][1]
                save_me = {
                    FoundPart(
                        part=pick,
                        chip_label=pick.part_no,
                        prefix="",
                        suffix="",
                        family="",
                    ): 1
                }
                self.save_part(save_me, self.cur_image)
                self.locked_parts = save_me
                self.best_match = save_me
                self.ocr_stable.is_set()
                asyncio.get_event_loop().call_later(1, self.clear_search)
            case _:
                self.clear_search()

    async def ocr_task(self):
        await self.frame_ready.wait()
        while not asyncio.current_task().cancelled():
            try:
                await self.ocr_single_frame(self.cur_cv2_image)
            except Exception as e:
                logging.exception(e)
                self.detecting_state = f"Error: {e}"
            await asyncio.sleep(0.01)
        logging.error("OCR task cancelled")

    async def ocr_single_frame(self, frame):

        if self.hash_diff > 10 and not self.filters:
            self.add_flash_message("RESET", delay=5, color=(255, 0, 0))
            self.reset()

        if "Search" in self.features:
            self.detecting_state = "MANUAL"
            return

        bbox = self.bounding_boxes

        parts_found, self.raw_text = await self.ocr_frame()

        if "sticky" in self.features:
            if 1 <= count_matches(self.recently_found) < 8:
                left = self.recently_found.popleft()
                if len(left):
                    self.recently_found.insert(
                        max(1, len(self.recently_found)) - 5, left
                    )

        self.recently_found.append(parts_found)
        self.compute_similarity(frame)

        if self.ocr_stable.is_set():
            self.detecting_state = "ESC to reset, ENTER to save"

        if not parts_found and not self.locked_parts:
            recent = list(self.recently_found)
            if count_matches(recent[:-10]) == 0 and len(recent) > 10:
                self.detecting_state = "Searching for parts"
                self.best_match = None
                self.locked_parts = None
                self.locked_bbox = None
                self.ocr_stable.clear()

            elif (
                count_matches(recent) == 0 and len(recent) == self.recently_found.maxlen
            ):
                self.add_flash_message("Reset", delay=1, color=(255, 0, 0))
                self.reset()

        best_match, n_occ = self.find_best_match()
        if best_match:
            self.best_match = best_match
            self.found_part_event.set()
        if not self.best_match or n_occ <= 6:
            return

        # If we get here, and we're already stable, just look for more parts found
        if self.ocr_stable.is_set() and self.locked_parts is not None:
            if sum(self.locked_parts.values()) < sum(self.best_match.values()):
                self.locked_parts = self.best_match
                self.locked_bbox = bbox
                self.add_flash_message(
                    f"UPDATED LOCKED",
                    delay=2,
                    color=(0, 255, 0),
                    font=MEDIUM_FONT,
                )
            return

        if not self.best_match and not self.locked_parts:
            self.ocr_stable.clear()
            self.detecting_state = "No parts found"
            return

        self.ocr_stable.set()
        self.locked_parts = self.best_match
        self.locked_bbox = bbox

        logging.info(f"LOCKED {self.best_match.keys()}")
        self.add_flash_message(
            f"LOCKED {" ".join(f"{part.chip_label} x {self.best_match[part]}" for part in self.best_match.keys())}",
            delay=2,
            color=(0, 255, 0),
            font=MEDIUM_FONT,
        )

    async def ocr_frame(self) -> tuple[defaultdict[Any, int], str]:

        self.ocr_req_queue.put(self.cur_cv2_image)
        result = await asyncio.to_thread(self.ocr_res_queue.get)
        self.counter.update(["ocr"])

        all_text = " ".join([text for text, confidence, bounding_box in result])

        # Find and count all parts in the OCR results
        matched_parts = defaultdict(int)
        bboxes: list[BoundingBox] = []

        do_disambiguate = (
            count_matches(self.recently_found) == 0
            and len(self.recently_found) == self.recently_found.maxlen
        )

        try:
            noted_prefix = ""
            for value, confidence, position in result:
                value = value.strip()
                if value in ("LM", "LN", "TL"):
                    noted_prefix = value
                    continue
                value = re.sub(r"[^a-zA-Z0-9/-]", "", value)
                if part := self.catalog.lookup_part_from_text(
                    noted_prefix + value.strip(), do_disambiguate=do_disambiguate
                ):
                    if part.part:
                        matched_parts[part] += 1
                bboxes.append(
                    BoundingBox(
                        confidence,
                        value,
                        ocrmac.convert_coordinates_pil(
                            position, self.frame_width, self.frame_height
                        ),
                        bool(part and part.part),
                    )
                )
        except Exception as e:
            logging.exception(e)
            self.add_flash_message(f"Error: {e}", delay=5, color=(255, 0, 0))

        self.bounding_boxes = bboxes
        return matched_parts, all_text

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

    def find_best_match(self) -> tuple[None, int] | tuple[dict[FoundPart, int], int]:
        if len(self.recently_found) < 1:
            return None, 0

        # Find the best frame by number of occurrences of parts
        sortable = [
            (
                aggregate(frame).total(),
                -len(frame),
                tuple(frame.keys()),
                frame,
            )
            for frame in self.recently_found
        ]
        counts = Counter(frame[2] for frame in sortable)
        ordered = list(sorted(sortable, key=lambda x: x[:2], reverse=True))
        return ordered[0][3], counts[ordered[0][2]]

    def draw_ui(self, ui: Image.Image):
        # Draw UI
        # - Frame number
        # - Currently found parts
        # - Last status message for its visible duration
        # - Last (stable) bounding boxes
        ui = ui.copy()
        drawer = ImageDraw.Draw(ui, "RGB")

        if self.verbose:
            self.draw_verbose(drawer)
        if "No Boxes" not in self.features:
            self.draw_bboxes(drawer)
        self.draw_status_bar(drawer)
        self.display_found_chips(drawer)
        if "Search" in self.features:
            self.draw_search_results(drawer)
        # Flash messages mid-screen
        self.draw_flash_messages(drawer)

        return ui

    def draw_verbose(self, drawer):
        drawer.text(
            (10, 880),
            self.raw_text,
            (192, 192, 128),
            FIXED_FONT,
            anchor="lm",
            stroke_width=1,
            font_size=12,
        )
        text = [
            "  ".join([f"{g:<12s}" for g in group])
            for group in itertools.batched(
                unique_everseen(
                    sorted(
                        [i for i in self.catalog.recent_lookups if i],
                        key=lambda x: len(x),
                    )
                ),
                12,
            )
        ]
        drawer.multiline_text(
            (10, 900), "\n".join(text), (64, 96, 96), font=FIXED_FONT, font_size=10
        )

    def draw_search_results(self, drawer):
        if time.time() % 1 < 0.5:
            cursor = True
        else:
            cursor = False
        drawer.text(
            (200, 475),
            "SEARCH: " + self.key_buffer + ("|" if cursor else ""),
            (0, 255, 255) if not "Action" in self.features else (255, 95, 0),
            font=FIXED_FONT,
            stroke_width=1,
            fill_width=1.2,
        )
        # count = len(self.search_results)
        drawer.text(
            (600, 475),
            f"page {self.search_result_page}",
            (96, 255, 64),
            font=FIXED_FONT,
            stroke_width=1,
            fill_width=1.1,
        )
        top = 600
        for n, (text, part, color) in enumerate(
            self.search_results[:NUM_SEARCH_RESULTS]
        ):
            drawer.multiline_text(
                (200, top),
                text,
                color,
                font=FIXED_FONT,
                stroke_fill=color,
                stroke_width=0.5,
            )
            bbox = drawer.multiline_textbbox(
                (200, top),
                text,
                font=FIXED_FONT,
            )
            top = bbox[3] + 1
        if not self.search_results:
            drawer.text(
                (200, 500),
                "No matches",
                (255, 64, 0),
                font=FIXED_FONT,
                stroke_width=0,
            )

    def draw_flash_messages(self, drawer):
        top = int(self.frame_width // 2), int(self.frame_height // 4)
        for line in self.flash_messages:
            size = drawer.textbbox(
                top, line.text, font=line.font or MAIN_FONT, spacing=1.25
            )
            drawer.text(
                (size[0], size[1]),
                fill=line.color,
                text=line.text,
                font=line.font,
                align="left",
                anchor="mm",
            )
            top = (top[0], size[3] + 5)
        self.flash_messages = [
            msg for msg in self.flash_messages if msg.expire > datetime.now()
        ]

    def display_found_chips(self, drawer):
        if not (self.best_match or self.locked_parts):
            return

        use = self.locked_parts or self.best_match
        if self.locked_parts:
            color = (255, 255, 255)
        else:
            color = (128, 128, 128)
        top = 50
        args = dict(
            xy=(50, top),
            font=MATCH_FONT,
            anchor="la",
            stroke_width=1,
        )
        for n, (part, n_occ) in enumerate(use.items()):
            # try:
            #     pins = int(part.part.pins)
            # except (AttributeError, ValueError, TypeError):
            #     pins = -1
            text = format_part_info(n_occ, part.chip_label, part.part)
            # size_args = args.copy()
            # del size_args["fill"]
            size = drawer.multiline_textbbox(**args, text=text)
            drawer.multiline_text(**args, text=text, fill=color)
            args["xy"] = (50, 2 + size[3])

    def draw_status_bar(self, drawer):
        # Status line
        for kwargs in (
            {
                "x": 10,
                **(
                    {"text": "LOCKED", "fill": (0, 255, 0)}
                    if self.ocr_stable.is_set()
                    else {"text": "SEARCHING", "fill": (192, 0, 0)}
                ),
            },
            {"y": 0, "x": 300, "text": f"{self.detecting_state[:70]}"},
            {
                "x": 10,
                "y": -30,
                "text": " ".join(
                    [f"{k}={n}" for k, n in self.counter.items() if k not in ("frame",)]
                ),
                "fill": 0,
                "stroke_width": 0.5,
                "stroke_fill": 0x666666,
            },
            {
                "x": 1000,
                "text": f"OCR {''.join('R' if s.is_alive() else 'x' for s in self.ocr_pool)}",
            },
            {
                "x": 1000,
                "y": -35,
                "text": f"{' '.join(self.features.keys())}",
                "fill": (0, 96, 255),
            },
            {
                "x": 1200,
                "text": f"SIM {self.hash_diff or 0:1.2f}",
                "fill": 0xFF0000 if self.hash_diff > 8 else 0x00FF00,
            },
            {
                "x": 1400,
                "text": f"MAT {count_matches(self.recently_found):3d}",
            },
            {"x": 1500, "text": f"BUF {len(self.recently_found):03d}"},
            {"x": 1700, "text": f"{self.counter['frame']:06d}"},
            {"x": 1800, "text": f"{datetime.now().strftime('%H:%M:%S')}"},
            {
                "": 600,
                "y": -55,
                "text": "PREV: "
                + "/".join(
                    format_part_info(
                        qty,
                        part.chip_label,
                        part.part,
                        columns=("n_occ", "part_no", "part_string"),
                    ).replace(" | ", " x ", 1)
                    for part, qty in (self.previous_save or {}).items()
                ),
            },
        ):
            x = kwargs.get("x", 10)
            y = kwargs.get("y", 0)
            args = {
                # "stroke_fill": (0, 0, 0, 50),
                "stroke_fill": kwargs.get(
                    "stroke_fill", kwargs.get("fill", (255, 255, 255))
                ),
                "fill": kwargs.get("fill", (255, 255, 255)),
                "anchor": "lm",
                "font": kwargs.get("font", FIXED_FONT),
                "stroke_width": 1,
                "fill_width": 1,
            } | kwargs
            drawer.text((x, self.frame_height - 20 + y, 0), **args)
            # logging.info("%s", f"{x} {text}, {arg}")

    def draw_bboxes(self, drawer):
        bboxes = self.get_drawable_bboxes()
        for box in bboxes:
            match box:
                case BoundingBox(confidence, _, _, True) if confidence < 0.5:
                    color = (128, 0, 0)
                case BoundingBox(confidence, _, _, True) if 0.5 <= confidence < 1:
                    color = (192, 192, 192)
                case BoundingBox(_, _, _, True):
                    color = (96, 128, 96)
                case BoundingBox(confidence, _, _, False) if confidence < 0.5:
                    color = (64, 0, 0, 64)
                case BoundingBox(confidence, _, _, False) if confidence < 0.8:
                    color = (128, 0, 0, 96)
                case BoundingBox(_, _, _, False):
                    color = (128, 0, 0, 128)
                case _:
                    color = (0, 32, 32)
            drawer.rectangle(box.position, outline=color, width=1 + int(box.matched))
            drawer.text(
                (box.position[0] + 3, box.position[1]),
                f"{box.text.strip()}",
                (255, 255, 255, 128 if box.matched else 255),
                MEDIUM_FONT,
                anchor="lb",
                stroke_width=0.8,
            )
            drawer.text(
                (box.position[2], box.position[3] + 2),
                f"{box.confidence * 100:3.0f}",
                color,
                SMALL_FONT,
                anchor="rt",
                stroke_width=0.4,
            )

    def get_drawable_bboxes(self):
        bboxes = self.locked_bbox or self.bounding_boxes
        if self.locked_bbox:
            # Filter out bboxes other than the matching ones
            bboxes = [
                box
                for box in self.locked_bbox
                if box.matched
                or box.text.strip().upper()
                in [p.chip_label for p in self.locked_parts.keys()]
            ]
        return bboxes

    def reset(self):
        # self.frame_ready.clear()
        self.ocr_stable.clear()
        self.recently_found.clear()
        self.best_match = None
        self.detecting_state = "Searching for parts"
        self.flash_messages.clear()
        self.locked_parts = None
        self.locked_bbox = None
        self.filters.clear()
        for timer in self.features.values():
            if isinstance(timer, TimerHandle):
                timer.cancel()
        self.features.clear()
        self.alphabeta.reset()
        self.clear_search()

    async def main_loop(self):
        lv = asyncio.create_task(self.live_video(), name="live_video")
        ui = asyncio.create_task(self.ui_task(), name="ui_task")
        ocr = asyncio.create_task(self.ocr_task(), name="ocr_task")

        res = await ui
        ocr.cancel()
        lv.cancel()
        await ocr
        await lv
        self.cam.release()
        cv2.destroyAllWindows()
        for worker in self.ocr_pool:
            worker.terminate()

    def save_part(self, locked_parts, ui: Image.Image):
        self.previous_save = self.locked_parts
        fn = self.catalog.save_part(
            ui,
            self.cur_image,
            self.opath,
            locked_parts,
            self.raw_text,
        )
        self.add_flash_message("SAVED", 5, (0, 255, 54), HUGE_FONT)
        self.detecting_state = f"SAVED to {fn}"

    def save_unknown(self):
        # Save with bounding boxes and text in a CSV
        fn = self.later_path / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        self.cur_image.save(fn, "PNG")
        with Path("try_later/index.json").open("a") as f:
            f.writelines(
                [
                    json.dumps(
                        {
                            "frame": self.counter["frame"],
                            "filename": str(fn),
                            "bounding_boxes": self.bounding_boxes,
                            "raw_text": self.raw_text,
                        }
                    ),
                    "\n",
                ]
            )

    async def force_lock(self):
        await self.add_feature("waiting")
        await self.found_part_event.wait()
        self.remove_feature("waiting")
        if self.best_match:
            await self.add_feature("force-locked")
            self.locked_parts = self.best_match
            self.locked_bbox = self.bounding_boxes
            self.ocr_stable.set()
        else:
            self.add_flash_message("No parts found to force-lock", 5, color=(255, 0, 0))
