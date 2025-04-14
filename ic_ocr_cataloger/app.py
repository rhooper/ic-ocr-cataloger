import asyncio
import itertools
import json
import logging
from asyncio import Event, TimerHandle
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from more_itertools import unique_everseen
from PIL import Image, ImageDraw

from .camera import Camera
from .catalog import Catalog
from .filters import ColorMapFilter
from .fonts import (
    BIG_FONT,
    FIXED_FONT,
    HUGE_FONT,
    MAIN_FONT,
    MATCH_FONT,
    MEDIUM_FONT,
    SMALL_FONT,
)
from .keyhandler import KeyEvent, KeyMap
from .keys.main import (
    MainBrightnessContrastKey,
    MainForceLockKey,
    MainGaussianBlurKey,
    MainGrayscaleKey,
    MainInvertKey,
    MainQuitKey,
    MainReimportKey,
    MainResetKey,
    MainSaveKey,
    MainSearchForLaterKey,
    MainSearchKey,
    MainSharpenKey,
    MainStickyKey,
    MainVerboseKey,
)
from .models import BoundingBox
from .ocr import OCR
from .search import SENTINEL, Search
from .ui import FlashMessage
from .util import count_matches, format_part_info

SEARCH_FLAG = "Search"


class ColorMapKey(KeyEvent):
    keys = b"u"

    def __init__(self, app: "App", context):
        super().__init__(app, context)
        self.color_map = ColorMapFilter()

    async def triggered(self, key):
        if self.color_map not in self.app.cam.filters:
            self.app.cam.filters.append(self.color_map)
        else:
            self.color_map.next_color_map()


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
        self.default_key_handler = KeyMap(
            MainResetKey(self, None),
            MainQuitKey(self, None),
            MainReimportKey(self, None),
            MainSaveKey(self, None),
            ColorMapKey(self, None),
            MainGrayscaleKey(self, None),
            MainInvertKey(self, None),
            MainGaussianBlurKey(self, None),
            MainSharpenKey(self, None),
            MainBrightnessContrastKey(self, None),
            MainStickyKey(self, None),
            MainForceLockKey(self, None),
            MainSearchKey(self, None),
            MainSearchForLaterKey(self, None),
            MainVerboseKey(self, None),
        )
        self.key_handler = self.default_key_handler
        self.args = args
        self.later_path = Path(
            Path(config["main"].get("try_later_dir", "ic_try_later")).expanduser()
        )
        self.later_path.mkdir(exist_ok=True, parents=True)
        self.opath = Path(
            Path(config["main"].get("saved_images_dir", "ic_saved")).expanduser()
        )
        self.catalog = Catalog(config)
        self.config = config

        self.counter = Counter(n=self.catalog.get_max(), frame=0)
        self.detecting_state = "Searching for parts"

        self.found_part_event = Event()

        self.flags: dict[str, TimerHandle | None] = {}
        self.flash_messages: deque[FlashMessage] = deque(maxlen=3)

        self.opath.mkdir(exist_ok=True, parents=True)
        self.previous_save = None
        self.verbose = False

        self.cam = Camera(self, config)
        self.ocr = OCR(self, ocr_req_queue, ocr_res_queue)
        self.search = Search(self, self.catalog)

    async def set_flag(
        self, flag: str, timeout: int = 15, timeout_callback: Callable | None = None
    ) -> None:
        # if "flag" in self.features:
        #     self.remove_feature(flag)
        if timeout > 0:
            self.flags[flag] = asyncio.get_event_loop().call_later(
                timeout,
                lambda: [
                    self.clear_flag(flag),
                    timeout_callback and timeout_callback(flag),
                ],
            )
        else:
            self.flags[flag] = None

    def clear_flag(self, flag: str):
        timer: TimerHandle | None = self.flags.get(flag, SENTINEL)
        if timer is SENTINEL:
            return
        if isinstance(timer, TimerHandle):
            timer.cancel()
        del self.flags[flag]

    def add_flash_message(
        self, message, delay=15, color=(255, 255, 255), font=BIG_FONT
    ):
        if message not in self.flash_messages:
            flash_message = FlashMessage(
                datetime.now() + timedelta(seconds=delay), message, color, font=font
            )
            self.flash_messages.append(flash_message)
            return flash_message

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

            if self.cam.frame_ready.is_set():
                self.cam.frame_ready.clear()
                ui = self.draw_ui(self.cam.cur_image)
                ui_im = cv2.cvtColor(np.array(ui), cv2.COLOR_RGB2BGR)
                cv2.imshow("UI", ui_im)

            key = cv2.waitKey(1)

            await self.key_handler.press(key)

    async def handle_save(self):
        ui = self.cam.cur_image.copy()
        if self.ocr.locked_parts:
            self.save_part(
                self.ocr.locked_parts,
                ui,
            )
        else:
            self.add_flash_message(
                "No parts locked (! to force-lock)",
                delay=1,
                color=(0, 255, 255),
            )

    async def do_reimport(self):
        self.add_flash_message(
            "Reimporting catalog",
            delay=2,
            color=(255, 255, 0),
            font=FIXED_FONT,
        )
        stats = await self.catalog.reimport()
        logging.info(stats)
        self.add_flash_message(
            f"Catalog loaded\n"
            + f'{stats["added"]} added, {stats["updated"]} updated, {stats["ignored"]} ignored\n'
            + f'{stats["line"]} lines => {stats["expanded"]} expanded parts\n'
            + f'{stats["files"]} files',
            delay=5,
            color=(255, 255, 0),
            font=FIXED_FONT,
        )
        # self.add_flash_message("Catalog Reloaded", delay=5, color=(255, 255, 0))

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
        if "No Boxes" not in self.flags:
            self.draw_bboxes(drawer)
        self.draw_status_bar(drawer)
        self.display_found_chips(drawer)

        if self.search.visible:
            self.search.draw_search_results(
                drawer,
            )
        # Flash messages mid-screen
        self.draw_flash_messages(drawer)

        return ui

    def draw_verbose(self, drawer):
        drawer.text(
            (10, 880),
            self.ocr.raw_text,
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

    def draw_flash_messages(self, drawer):
        top = int(self.cam.frame_width // 2), int(self.cam.frame_height // 4)
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
        if not (self.ocr.best_match or self.ocr.locked_parts):
            return

        use = self.ocr.locked_parts or self.ocr.best_match
        if self.ocr.locked_parts:
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
                    if self.ocr.ocr_stable.is_set()
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
                "text": f"OCR {''.join('R' if s.is_alive() else 'x' for s in self.ocr.ocr_pool or [])}",
            },
            {
                "x": 1000,
                "y": -35,
                "text": f"{' '.join(list(self.flags.keys()) + [self.key_handler.status_flag])}",
                "fill": (0, 96, 255),
            },
            {
                "x": 1200,
                "text": f"Δ {self.cam.hash_diff or 0:1.2f}",
                "fill": 0xFF0000 if self.cam.hash_diff > 8 else 0x00FF00,
            },
            {
                "x": 1400,
                "text": f"MAT {count_matches(self.ocr.recently_found):3d}",
            },
            {"x": 1500, "text": f"BUF {len(self.ocr.recently_found):03d}"},
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
            drawer.text((x, self.cam.frame_height - 20 + y, 0), **args)
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
        bboxes = self.ocr.locked_bbox or self.ocr.bounding_boxes
        if self.ocr.locked_bbox:
            # Filter out bboxes other than the matching ones
            bboxes = [
                box
                for box in self.ocr.locked_bbox
                if box.matched
                or box.text.strip().upper()
                in [p.chip_label for p in self.ocr.locked_parts.keys()]
            ]
        return bboxes

    def reset(self):
        # self.frame_ready.clear()
        self.ocr.reset()
        self.cam.reset()
        self.detecting_state = "Searching for parts"
        for timer in self.flags.values():
            if isinstance(timer, TimerHandle):
                timer.cancel()
        self.flash_messages.clear()
        self.flags.clear()
        self.search.reset()
        self.restore_keymap()

    async def main_loop(self):
        lv = asyncio.create_task(self.cam.live_video(), name="live_video")
        ui = asyncio.create_task(self.ui_task(), name="ui_task")
        self.ocr.run()
        ocr = asyncio.create_task(self.ocr.ocr_task(), name="ocr_task")
        if len(self.catalog.parts) == 0:
            self.add_flash_message(
                "No parts found in catalog.  Press r to load catalog",
                5,
                (255, 0, 0),
                font=BIG_FONT,
            )

        await ui

        # Shutdown initiated
        ocr.cancel()
        lv.cancel()
        await ocr
        await lv
        await self.cam.release()
        await self.ocr.stop()
        cv2.destroyAllWindows()

    def save_part(self, locked_parts, ui: Image.Image):
        self.previous_save = self.ocr.locked_parts
        fn = self.catalog.save_part(
            ui,
            self.cam.cur_image,
            self.opath,
            locked_parts,
            self.ocr.raw_text,
        )
        self.add_flash_message("SAVED", 5, (0, 255, 54), HUGE_FONT)
        self.detecting_state = f"SAVED to {fn}"

    def save_unknown(self):
        # Save with bounding boxes and text in a CSV
        fn = self.later_path / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        self.cam.cur_image.save(fn, "PNG")
        with (self.later_path / "try_later/index.json").open("a") as f:
            f.writelines(
                [
                    json.dumps(
                        {
                            "frame": self.counter["frame"],
                            "filename": str(fn),
                            "bounding_boxes": self.ocr.bounding_boxes,
                            "raw_text": self.ocr.raw_text,
                        }
                    ),
                    "\n",
                ]
            )

    def restore_keymap(self):
        self.activate_keymap(self.default_key_handler)

    def activate_keymap(self, keymap: KeyMap):
        self.key_handler = keymap
