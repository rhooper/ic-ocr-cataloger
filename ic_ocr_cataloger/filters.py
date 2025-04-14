from collections import deque

import cv2


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


class ColorMapFilter:
    def __init__(self):
        self._color_map = 0
        self.color_maps = deque(
            [
                cv2.COLORMAP_AUTUMN,
                cv2.COLORMAP_BONE,
                cv2.COLORMAP_JET,
                cv2.COLORMAP_WINTER,
                cv2.COLORMAP_RAINBOW,
                cv2.COLORMAP_OCEAN,
                cv2.COLORMAP_SUMMER,
                cv2.COLORMAP_SPRING,
                cv2.COLORMAP_COOL,
                cv2.COLORMAP_HSV,
                cv2.COLORMAP_PINK,
                cv2.COLORMAP_HOT,
                cv2.COLORMAP_PARULA,
                cv2.COLORMAP_MAGMA,
                cv2.COLORMAP_INFERNO,
                cv2.COLORMAP_PLASMA,
                cv2.COLORMAP_VIRIDIS,
                cv2.COLORMAP_CIVIDIS,
                cv2.COLORMAP_TWILIGHT,
                cv2.COLORMAP_TWILIGHT_SHIFTED,
                cv2.COLORMAP_TURBO,
                cv2.COLORMAP_DEEPGREEN,
            ]
        )

    def __call__(self, img):
        return cv2.applyColorMap(img, self.color_map)

    @property
    def color_map(self):
        return self.color_maps[0]

    def next_color_map(self):
        self.color_maps.rotate(-1)
