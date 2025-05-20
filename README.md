IC OCR Cataloger
----------------

IC OCR Cataloger is a Python package
that uses Optical Character Recognition (OCR)
to identify integrated circuits.
It includes a cataloging facilities using SQLite.

Installation
------------

This package exclusively uses [`pyproject.toml`](https://python-poetry.org/docs/pyproject/)
to manage dependencies and packaging.

To install the package, you can use `pip`:

```bash
pip install ic-ocr-cataloger
```
If your package installer does not work with pyproject.toml only projects, please file an issue.

This project uses a variety of fonts, which are currently hard-coded in `ic_ocr_cataloger/fonts.py`.
You can download the fonts from the following links:
https://www.nerdfonts.com/font-downloads and https://github.com/ryanoasis/nerd-fonts?tab=readme-ov-file
