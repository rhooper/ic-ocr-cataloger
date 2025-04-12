import asyncio
import configparser
import logging
import multiprocessing
from pathlib import Path

from .app import App

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="IC OCR Cataloger")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Location of config file",
        default=Path(__file__).parent / "default_config.ini",
    )
    return parser.parse_args()


def read_config(path: Path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def main():
    args = get_args()
    config = read_config(args.config)
    ocr_res_queue = multiprocessing.Queue(maxsize=2)
    ocr_req_queue = multiprocessing.Queue(maxsize=4)
    app = App(args, config, ocr_req_queue, ocr_res_queue)
    asyncio.gather(asyncio.run(app.main_loop()))


if __name__ == "__main__":
    main()
