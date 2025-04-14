import cv2

from ic_ocr_cataloger.fonts import SMALL_FONT
from ic_ocr_cataloger.keyhandler import KeyEvent
from ic_ocr_cataloger.ocr import STICKY


class MainResetKey(KeyEvent):
    keys = b"xX\x1b"

    async def triggered(self, key):
        self.app.reset()
        self.app.add_flash_message("Reset", delay=1, color=(255, 0, 0))


class MainQuitKey(KeyEvent):
    keys = b"qQ"

    async def triggered(self, key):
        self.app.add_flash_message("Quitting", delay=1, color=(255, 0, 0))
        self.app.cam.release()
        cv2.destroyAllWindows()
        self.app.ocr.stop()
        self.app.ocr.stop()
        raise KeyboardInterrupt()


class MainReimportKey(KeyEvent):
    keys = b"rR"

    async def triggered(self, key):
        await self.app.do_reimport()


class MainSaveKey(KeyEvent):
    keys = bytes([3, 13, 10])  # CTRL+C, ENTER, RETURN

    async def triggered(self, key):
        await self.app.handle_save()


class MainGrayscaleKey(KeyEvent):
    keys = b"gG"

    async def triggered(self, key):
        self.app.add_flash_message("Grayscale", 2, font=SMALL_FONT)
        self.app.cam.grayscale()


class MainInvertKey(KeyEvent):
    keys = b"iI"

    async def triggered(self, key):
        self.app.add_flash_message("Invert", 2, font=SMALL_FONT)
        self.app.cam.invert()


class MainGaussianBlurKey(KeyEvent):
    keys = b"lL"

    async def triggered(self, key):
        self.app.add_flash_message("Gaussian Blur", 2, font=SMALL_FONT)
        self.app.cam.blur()


class MainSharpenKey(KeyEvent):
    keys = b"Ss"

    async def triggered(self, key):
        self.app.add_flash_message("Sharpen", 2, font=SMALL_FONT)
        self.app.cam.sharpen()


class MainBrightnessContrastKey(KeyEvent):
    keys = b"bBcC"

    async def triggered(self, key):
        if chr(key).isupper():
            direction = -1
        else:
            direction = 1
        if key in b"cC":
            self.app.add_flash_message("Contrast", 2, font=SMALL_FONT)
            self.app.cam.contrast(value=direction)
        elif key in b"bB":
            self.app.add_flash_message("Brightness", 2, font=SMALL_FONT)
            self.app.cam.brightness(value=direction)


class MainStickyKey(KeyEvent):
    keys = b"."

    async def triggered(self, key):
        if STICKY in self.app.flags:
            self.app.clear_flag(STICKY)
        else:
            await self.app.set_flag(STICKY)


class MainForceLockKey(KeyEvent):
    keys = b"!"

    async def triggered(self, key):
        self.app.add_flash_message("Force Lock", 2, font=SMALL_FONT)
        await self.app.ocr.force_lock()


class MainSearchKey(KeyEvent):
    keys = b"/"

    async def triggered(self, key):
        self.app.search.activate()


class MainSearchForLaterKey(KeyEvent):
    keys = b">"

    async def triggered(self, key):
        self.app.add_flash_message("For Later", 4, color=(255, 255, 255))
        self.app.save_unknown()


class MainVerboseKey(KeyEvent):
    keys = b"vV"

    async def triggered(self, key):
        self.app.verbose = not self.app.verbose
        self.app.add_flash_message(f"Verbose: {self.app.verbose}", 2, font=SMALL_FONT)
