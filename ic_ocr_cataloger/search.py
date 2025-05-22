import asyncio
import re
import time
from enum import EnumType

from ic_ocr_cataloger.catalog import Catalog, FoundPart
from ic_ocr_cataloger.fonts import FIXED_FONT
from ic_ocr_cataloger.keyhandler import KeyEvent, KeyMap
from ic_ocr_cataloger.util import format_part_info

NO_BOXES = "No Boxes"
NUM_SEARCH_RESULTS = 16
SENTINEL = object()


class SearchKeyMap(KeyMap):
    name = "search"
    status_flag = "Search"

    def __init__(self, *handlers: KeyEvent, app=None, context=None):
        super().__init__(
            SearchEsc(app, context),
            SearchArrowKey(app, context),
            SearchDelBackspace(app, context),
            SearchSquareLeft(app, context),
            SearchSquareRight(app, context),
            SearchSemicolon(app, context),
            SearchApostrophe(app, context),
            SearchCharacterKey(app, context),
            *handlers,
            app=app,
            context=context,
        )

    async def press(self, key: int = None):
        result = await super().press(key)
        if isinstance(result, KeyMap):
            self.app.set_mode("search")
            self.app.activate_keymap(result)
        elif result:
            self.context.do_search()


class SearchSelectKeyMap(KeyMap):
    name = "search_select"
    status_flag = "Search>Select"

    def __init__(self, *handlers: KeyEvent, app=None, context=None):
        super().__init__(
            # Passthroughs:
            SearchDelBackspace(app, self),
            SearchSquareLeft(app, self),
            SearchSquareRight(app, self),
            SearchSemicolon(app, self),
            SearchApostrophe(app, self),
            # State-specific:
            ResultUpDownKey(app, self),
            ResultLeftRightKey(app, self),
            ResultUpDownKey(app, self),
            ResultSelectKey(app, self),
            ResultEnterKey(app, self),
            ResultEscapeKey(app, self),
            *handlers,
            app=app,
            context=context,
        )

    async def press(self, key: int = None):
        result = super().press(key)
        context = self.context
        if isinstance(result, KeyMap):
            context.context.set_mode("search_select")
            context.app.activate_keymap(result)
        elif result:
            context.do_search()


class SearchMode(EnumType):
    SEARCH = 1
    ACTION = 2


class ResultUpDownKey(KeyEvent):
    keys = (0, 1)

    async def triggered(self, key):
        context = self.context
        if key == 0:
            context.current_search_pick -= 1
        else:
            context.current_search_pick += 1
        if context.current_search_pick < 1:
            context.current_search_pick = len(context.search_results)
        elif context.current_search_pick > len(context.search_results):
            context.current_search_pick = 1


class ResultLeftRightKey(KeyEvent):
    keys = (2, 3)

    async def triggered(self, key):
        context = self.context
        pg = context.search_result_page
        if (
            len(context.search_results) < NUM_SEARCH_RESULTS
            and context.search_result_page == 0
        ):
            return

        if key == 2:
            context.search_result_page = min(0, context.search_result_page - 1)
        else:
            context.search_result_page += 1
        if context.search_result_page < 0:
            context.search_result_page = 0
        if pg != context.search_result_page:
            context.current_search_pick = 0


class ResultEscapeKey(KeyEvent):
    keys = 27

    async def triggered(self, key):
        context = self.context
        context.active_keymap = context.search_keymap
        context.current_search_pick = 0
        context.search_result_page = 0


class ResultSelectKey(KeyEvent):
    keys = b"1234567890abcdefABCDEF"

    async def triggered(self, key):
        self.context.current_search_pick = int(chr(key), 16)
        if self.context.current_search_pick == 0:
            self.context.current_search_pick = 10


class ResultEnterKey(KeyEvent):
    keys = (13, 10)

    async def triggered(self, key):
        context = self.context
        if not context.current_search_pick:
            return
        pick = context.search_results[context.current_search_pick - 1][1]
        save_me = {
            # TODO use common 74xx, 4000x, etc decoder
            FoundPart(
                part=pick,
                chip_label=pick.part_no,
                prefix="",
                suffix="",
                family="",
            ): 1
        }
        context.app.ocr_next_queue_item.locked_parts = save_me
        context.app.ocr_next_queue_item.best_match = save_me
        await self.app.save_part(save_me, context.cur_image)
        asyncio.get_event_loop().call_later(2, context.clear_search)


class Search:

    def __init__(self, app: "App", catalog: Catalog):
        self.app = app
        self.catalog = catalog
        self.visible = False

        self.search_keymap = SearchKeyMap(app=app, context=self)
        self.active_keymap = self.search_keymap
        self.select_keymap = SearchSelectKeyMap(app=app, context=self)

        self.search_string = "%"
        self.selected_search_result = None
        self.total_results = 0
        self.search_result_page = 0
        self.search_results = []
        self.current_search_pick = 0

    def activate(self):
        self.visible = True
        self.set_mode(SearchMode.SEARCH)
        self.app.activate_keymap(self.search_keymap)

    def deactivate(self):
        self.reset()

    def set_mode(self, mode):
        match mode:
            case SearchMode.SEARCH:
                self.active_keymap = self.search_keymap
                self.app.activate_keymap(self.search_keymap)
            case SearchMode.ACTION:
                self.active_keymap = self.select_keymap
                self.app.activate_keymap(self.select_keymap)

    @property
    def current_mode(self):
        if self.active_keymap == self.search_keymap:
            return SearchMode.SEARCH
        elif self.active_keymap == self.select_keymap:
            return SearchMode.ACTION
        else:
            return None

    def clear_search(self):
        self.active_keymap = self.search_keymap
        self.search_string = "%"
        self.search_result_page = 0
        self.selected_search_result = 0

    def do_search(self):
        self.search_results = []
        if len(self.search_string) > 2:
            if not (
                results := list(
                    self.catalog.search(
                        self.search_string,
                        limit=NUM_SEARCH_RESULTS,
                        offset=self.search_result_page * NUM_SEARCH_RESULTS,
                    )
                )
            ):
                if self.search_result_page > 0:
                    self.search_result_page = 0
                    results = list(
                        self.catalog.search(
                            self.search_string,
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

    def draw_search_results(self, drawer):
        if time.time() % 1 < 0.5:
            cursor = True
        else:
            cursor = False
        drawer.text(
            (200, 475),
            "SEARCH: " + self.search_string + ("|" if cursor else ""),
            (0, 255, 255) if self.active_keymap is self.select_keymap else (255, 95, 0),
            font=FIXED_FONT,
            stroke_width=1,
            fill_width=1.2,
        )
        # count = len(self.search_results)
        drawer.text(
            (600, 475),
            f"page {self.search_result_page + 1} of {self.total_results // NUM_SEARCH_RESULTS + 1}",
            (96, 255, 64),
            font=FIXED_FONT,
            stroke_width=1,
            fill_width=1.1,
        )
        top = 600
        for n, (text, part, color) in enumerate(self.search_results):
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

    def reset(self):
        self.visible = False
        self.active_keymap = None
        self.app.restore_keymap()
        self.search_string = "%"
        self.search_result_page = 0
        self.selected_search_result = 0
        self.current_search_pick = 0
        self.search_results = []


class SearchEsc(KeyEvent):
    keys = [27]

    async def triggered(self, key):
        self.context.search_string = "%"
        self.app.restore_keymap()


class SearchArrowKey(KeyEvent):
    keys = (
        0,
        1,
        2,
        3,
        13,
    )  # UP, DOWN, LEFT, RIGHT, ENTER

    async def triggered(self, key):
        self.context.set_mode(SearchMode.ACTION)
        self.context.select_keymap.press(key)


class SearchDelBackspace(KeyEvent):
    keys = (8, 127)  # DEL, BACKSPACE

    async def triggered(self, key):
        self.context.search_string = self.context.search_string[:-1]


class SearchSquareLeft(KeyEvent):
    keys = b"["

    async def triggered(self, key):
        if self.context.search_string.startswith("%"):
            self.context.search_string = self.context.search_string[1:]
        else:
            self.context.search_string = "%" + self.context.search_string.lstrip("%")


class SearchSquareRight(KeyEvent):
    keys = b"]"

    async def triggered(self, key):

        self.context.search_string = (
            self.context.search_string[:-1]
            if self.context.search_string.endswith("%")
            and not self.context.search_string.startswith("%")
            else self.context.search_string + "%"
        )


class SearchSemicolon(KeyEvent):
    keys = b";"

    async def triggered(self, key):
        self.context.search_string = re.sub(
            r"^(.*(\d{2}))%*(.*)$", r"%\1%\2", self.context.search_string
        )


class SearchApostrophe(KeyEvent):
    keys = b"'"

    async def triggered(self, key):
        self.context.search_string = re.sub(
            r"^[A-Z%]*", "%", self.context.search_string
        )


class SearchCharacterKey(KeyEvent):
    keys = bytes(range(32, 127))

    async def triggered(self, key):
        self.context.search_string += chr(key).upper()
        if self.context.search_string[-2:].startswith(
            "%"
        ) and not self.context.search_string.startswith("%"):
            self.context.search_string = (
                self.context.search_string[:-2] + self.context.search_string[-1] + "%"
            )
