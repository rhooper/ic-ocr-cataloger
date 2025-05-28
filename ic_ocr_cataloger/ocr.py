import abc
import asyncio
import logging
import multiprocessing
import re
import time
from abc import abstractmethod
from asyncio import Event
from collections import defaultdict, deque

from PIL import Image

from ic_ocr_cataloger.camera import Camera
from ic_ocr_cataloger.catalog import FoundPart
from ic_ocr_cataloger.fonts import MEDIUM_FONT
from ic_ocr_cataloger.models import BoundingBox
from ic_ocr_cataloger.util import count_matches, evaluate_best_match

STICKY = "sticky"


class OCREngine(abc.ABC):

    @abstractmethod
    def recognize(self, image) -> tuple:
        pass


class OCRMacEngine(OCREngine):
    def __init__(
        self,
        frame_width=0,
        frame_height=0,
        *args,
        language_preference=("en-US",),
        framework="vision",
        recognition_level="accurate",
        **extra_kwargs,
    ):
        from ocrmac import ocrmac

        self.engine = ocrmac
        self.args = args
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.kwargs = {
            **extra_kwargs,
            "language_preference": [*language_preference],
            "framework": framework,
            "recognition_level": recognition_level,
        }

    def recognize(
        self, image
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        return self.engine.OCR(
            image,
            *self.args,
            **self.kwargs,
        ).recognize(px=True)


class TesseractOcrEngine:
    def __init__(self):
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "Tesseract OCR engine not found. Please install pytesseract."
            )

        self.engine = pytesseract

    def recognize(self, image) -> tuple:
        res = self.engine.image_to_data(
            image, config="--psm 12", lang="eng", output_type=self.engine.Output.DICT
        )
        return [
            (
                res["text"][i],
                res["conf"][i],
                (
                    res["left"][i],
                    res["top"][i],
                    res["left"][i] + res["width"][i],
                    res["left"][i] + res["height"][i],
                ),
            )
            for word, left, top, width, left in zip()
            if int(res["conf"][i]) > 0
        ]


class OCR:
    def __init__(self, app, ocr_req_queue, ocr_res_queue):
        self.ocr_pool = None
        self.app = app
        self.cam: Camera = app.cam
        self.locked_parts: dict[FoundPart, int] = {}
        self.locked_bbox = None
        self.ocr_res_queue: multiprocessing.Queue = ocr_res_queue
        self.last_ocr_result_timestamp = 0
        self.ocr_stable = Event()
        self.raw_text: str = ""
        self.recently_found: deque[dict[FoundPart, int]] = deque(maxlen=50)
        self.ocr_req_queue: multiprocessing.Queue = ocr_req_queue
        self.bounding_boxes: list[BoundingBox] = []
        self.best_match: dict[FoundPart, int] | None = None

    def start_workers(self):
        engine_name = self.app.config.get("main", "engine")

        self.ocr_pool = [
            OCRWorker(
                n,
                self.ocr_req_queue,
                self.ocr_res_queue,
                self.cam.frame_width,
                self.cam.frame_height,
                engine_name=engine_name,
                shutdown_pending=self.app.shutdown_pending,
            )
            for n in range(self.app.config["main"].getint("ocr_workers", 1))
        ]

        for worker in self.ocr_pool:
            worker.start()

    async def ocr_result_loop(self):
        await self.cam.frame_ready.wait()
        while not asyncio.current_task().cancelled():
            try:
                await self.ocr_single_frame(self.cam.cur_cv2_image)
            except Exception as e:
                logging.exception(e)
                self.app.detecting_state = f"Error: {e}"
            await asyncio.sleep(0.01)
        logging.error("OCR task cancelled")

    async def ocr_single_frame(self, frame):

        bbox = self.bounding_boxes

        parts_found, self.raw_text = await self.ocr_frame()

        no_prefix = set(
            [part.chip_label[len(part.prefix) :] for part in parts_found.keys()]
        )
        if len(no_prefix) == 1 and len(parts_found) > 1:
            # Combine same part with different prefix, using whichever is first
            parts = list(parts_found.items())
            for rest in parts[1:]:
                parts_found[parts[0][0]] += rest[1]
                del parts_found[rest[0]]

        if STICKY in self.app.flags:
            if 1 <= count_matches(self.recently_found) < 8:
                left = self.recently_found.popleft()
                if len(left):
                    self.recently_found.insert(
                        max(1, len(self.recently_found)) - 5, left
                    )

        self.recently_found.append(parts_found)

        if self.ocr_stable.is_set():
            self.app.detecting_state = "ESC to reset, ENTER to save"

        if not parts_found and not self.locked_parts:
            recent = list(self.recently_found)
            if count_matches(recent[:-10]) == 0 and len(recent) > 10:
                self.app.detecting_state = "Searching for parts"
                self.best_match = None
                self.locked_parts = None
                self.locked_bbox = None
                self.ocr_stable.clear()

            elif (
                count_matches(recent) == 0 and len(recent) == self.recently_found.maxlen
            ):
                self.app.reset()

        best_match, n_occ = self.find_best_match()
        if best_match:
            self.best_match = best_match
            self.app.found_part_event.set()
        if not self.best_match or n_occ <= 6:
            return

        # If we get here, and we're already stable, just look for more parts found
        if self.ocr_stable.is_set() and self.locked_parts is not None:
            if sum(self.locked_parts.values()) < sum(self.best_match.values()):
                self.locked_parts = self.best_match
                self.locked_bbox = bbox
                self.app.add_flash_message(
                    f"UPDATED LOCKED",
                    delay=2,
                    color=(0, 255, 0),
                    font=MEDIUM_FONT,
                )
            return

        if not self.best_match and not self.locked_parts:
            self.ocr_stable.clear()
            self.app.detecting_state = "No parts found"
            return

        self.ocr_stable.set()
        self.locked_parts = self.best_match
        self.locked_bbox = bbox

        logging.info(f"LOCKED {self.best_match.keys()}")
        self.app.add_flash_message(
            f"LOCKED {" ".join(f"{part.chip_label} x {self.best_match[part]}" for part in self.best_match.keys())}",
            delay=2,
            color=(0, 255, 0),
            font=MEDIUM_FONT,
        )

    async def ocr_frame(
        self,
    ) -> tuple[defaultdict[FoundPart, int] | dict[FoundPart, int], str]:

        self.ocr_req_queue.put(self.cam.cur_cv2_image)
        timestamp, result = await asyncio.to_thread(self.ocr_res_queue.get)
        if timestamp < self.last_ocr_result_timestamp:
            return {}, ""
        self.last_ocr_result_timestamp = timestamp

        self.app.counter.update(["ocr"])
        all_text = " ".join([text for text, confidence, bounding_box in result])

        # Find and count all parts in the OCR results
        matched_parts = defaultdict(int)
        bboxes: list[BoundingBox] = []

        do_disambiguate = (
            count_matches(self.recently_found) == 0
            and len(self.recently_found) == self.recently_found.maxlen
        )

        # TODO ensure ON suffixes are discarded (Eg MC74HC280N mathcing 7428)
        # Improve 74-series matching

        try:
            noted_prefix = ""
            for value, confidence, positions in result:
                value = value.strip()
                if value in ("LM", "LN", "TL", "DS"):
                    noted_prefix = value
                    continue
                value = re.sub(r"[^a-zA-Z0-9/-]", "", value)
                if part := self.app.catalog.lookup_part_from_text(
                    noted_prefix + value.strip(), do_disambiguate=do_disambiguate
                ):
                    if part.part:
                        matched_parts[part] += 1
                bboxes.append(
                    BoundingBox(
                        confidence,
                        value,
                        positions,
                        bool(part and part.part),
                    )
                )
        except Exception as e:
            logging.exception(e)
            self.app.add_flash_message(f"Error: {e}", delay=5, color=(255, 0, 0))

        self.bounding_boxes = bboxes
        return matched_parts, all_text

    def find_best_match(self) -> tuple[None, int] | tuple[dict[FoundPart, int], int]:
        if len(self.recently_found) < 1:
            return None, 0

        return evaluate_best_match(self.recently_found)

    async def force_lock(self):
        if self.best_match:
            await self.app.set_flag("force-locked")
            self.locked_parts = self.best_match
            self.locked_bbox = self.bounding_boxes
            self.ocr_stable.set()
        else:
            self.app.add_flash_message(
                "No parts found to force-lock", 5, color=(255, 0, 0)
            )

    def reset(self):
        self.ocr_stable.clear()
        self.locked_parts = {}
        self.locked_bbox = None
        self.best_match = None
        self.recently_found.clear()
        self.bounding_boxes.clear()
        self.raw_text = ""
        self.app.detecting_state = "Searching for parts"

    def stop(self):
        for worker in self.ocr_pool:
            logging.info("Stopping OCR worker %s", worker.name)
            worker.terminate()
            worker.join()
        self.ocr_req_queue.close()
        self.ocr_res_queue.close()
        logging.info("Stopped OCR workers and queues closed")


class OCRWorker(multiprocessing.Process):
    """
    Worker process to run (multiple) OCR workers.
    """

    def __init__(
        self,
        n,
        req_queue,
        res_queue,
        frame_width,
        frame_height,
        engine_name: str = "ocrmac",
        shutdown_pending: Event = None,
    ):
        super().__init__(daemon=True)
        self.n = n
        self.req_queue = req_queue
        self.res_queue = res_queue
        self.daemon = True
        self.name = f"ocr_worker_{n}"
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.engine_name = engine_name
        self._engine: OCREngine | None = None
        self.shutdown_event = shutdown_pending

    def run(self):
        logging.info("%s", f"OCR Worker no {self.n} ready")

        match self.engine_name:
            case "tesseract":
                self._engine = TesseractOcrEngine()
            case "ocrmac" | _:
                self._engine = OCRMacEngine(
                    frame_width=self.frame_width, frame_height=self.frame_height
                )

        while not self.shutdown_event.is_set():
            try:
                self.ocr_next_queue_item()
            except (KeyboardInterrupt, multiprocessing.ProcessError):
                logging.info("Keyboard interrupt")
                self.res_queue.close()
                break
            except Exception as e:
                logging.exception(e)
                time.sleep(1)

    def ocr_next_queue_item(self):
        frame = self.req_queue.get()
        if frame is None:
            return
        im = Image.fromarray(frame, "RGB")
        if im is None:
            return
        result = self._engine.recognize(im)
        if self.res_queue.full():
            logging.info("Waiting for result queue")
        self.res_queue.put((time.time(), result))
