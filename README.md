# PhotoEditor

A simple GUI photo editor built with OpenCV. Adjust the image live using trackbars (sliders).

## Features

- Adjust brightness
- Adjust contrast
- Convert to grayscale
- Apply thresholding (binary or inverse)
- Add padding

## Requirements

- Python 3
- OpenCV (`opencv-python`)
- NumPy

Install with:

```bash
pip install opencv-python numpy
```

## Usage

Put an image named `cat.jpg` in the same folder (or change the filename in `photo_editor.py`), then run:

```bash
python photo_editor.py
```

Move the sliders to edit the image.

- Press `s` to save the current image as `edited.jpg`
- Press `q` to quit
