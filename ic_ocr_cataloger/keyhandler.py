import abc
import asyncio
import logging
from abc import abstractmethod
from typing import Iterable


class KeyEvent(abc.ABC):
    __match_args__ = ("keys",)
    keys: bytes | Iterable[int] = b""
    help = ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_names()})"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.keys is None:
            raise ValueError(f"{cls.__name__} keys cannot be None")
        if isinstance(cls.keys, str):
            cls.keys = cls.keys.encode("utf-8")
            logging.warning(
                "KeyEvent %s keys should be bytes or iterable of int, not str",
                cls.__name__,
            )
        elif isinstance(cls.keys, int):
            cls.keys = bytes([cls.keys])

    def to_names(self):
        names = []
        for char in self.keys:
            if 32 <= char <= 126:
                names.append(chr(char))
            else:
                match char:
                    case 0:
                        names.append("UP")
                    case 1:
                        names.append("DOWN")
                    case 2:
                        names.append("LEFT")
                    case 3:
                        names.append("RIGHT")
                    case 8:
                        names.append("BACKSPACE")
                    case 9:
                        names.append("TAB")
                    case 13:
                        names.append("ENTER")
                    case 27:
                        names.append("ESCAPE")
                    case 127:
                        names.append("DELETE")
                    case _:
                        names.append(f"'{char}'")
        return ", ".join(f"<{name}>" for name in names)

    def __init__(self, app, context):
        self.context = context
        self.app = app

    async def __call__(self, key: int, context=None):
        if context:
            self.context = context
        if key in self.keys:
            return await self.triggered(key)

    @abstractmethod
    async def triggered(self, key):
        pass


class KeyMap:
    name = "KeyMap"
    status_flag = ""

    def __init__(self, *handlers: KeyEvent, app=None, context=None):
        self.context = context or {}
        self.app = app

        if not handlers:
            return
        self.handlers = list(reversed(handlers))
        for handler in self.handlers:
            if not isinstance(handler.keys, (Iterable, bytes)):
                raise TypeError(f"{type(handler)} keys {handler.keys} are not iterable")
        self.keymap = {
            key: handler for handler in self.handlers for key in handler.keys
        }

    async def press(self, key):
        handler = self.keymap.get(key)
        if handler:
            return await handler(key) or True
        else:
            await asyncio.sleep(0.05)
        return False


class MainKeyMap(KeyMap):
    name = "main"
    status_flag = ""
