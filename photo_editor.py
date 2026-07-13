"""
Mini Photo Editor - GUI version with OpenCV trackbars (sliders).

Move the sliders and the image updates live (via a callback function).
Keys:  s = save the current image    q = quit
"""

import cv2
import numpy as np


# --- Each operation is its own small function (modular) --------------------
def apply_brightness(img, value):
    # value goes from -100 to 100, added to every pixel
    result = img.astype(np.int16) + value
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast(img, factor):
    # factor multiplies every pixel (1.0 = no change)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # back to 3 channels


def apply_threshold(img, value, inverse):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, value, 255, mode)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def apply_padding(img, size):
    return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT)


# The image we load and the window name (used by the callback)
original = cv2.imread("cat.jpg")   # change this to your image file
window = "Photo Editor"
result = original                   # holds the latest edited image (for saving)


# --- Callback: runs every time ANY slider moves ----------------------------
def update(x):
    global result

    # Read the current slider positions
    brightness = cv2.getTrackbarPos("Brightness", window) - 100
    contrast = cv2.getTrackbarPos("Contrast", window) / 10
    grayscale = cv2.getTrackbarPos("Grayscale", window)
    thresh_on = cv2.getTrackbarPos("Threshold On", window)
    thresh_val = cv2.getTrackbarPos("Threshold", window)
    inverse = cv2.getTrackbarPos("Inverse", window)
    padding = cv2.getTrackbarPos("Padding", window)

    # Apply the operations one after another
    result = original.copy()
    result = apply_brightness(result, brightness)
    result = apply_contrast(result, contrast)
    if grayscale:
        result = apply_grayscale(result)
    if thresh_on:
        result = apply_threshold(result, thresh_val, inverse)
    if padding:
        result = apply_padding(result, padding)

    cv2.imshow(window, result)


# --- Main program ----------------------------------------------------------
def main():
    cv2.namedWindow(window)

    # Create the sliders. The last argument is the callback that runs on change.
    cv2.createTrackbar("Brightness", window, 100, 200, update)  # 0-200 -> -100..100
    cv2.createTrackbar("Contrast", window, 10, 30, update)      # 0-30  -> 0.0..3.0
    cv2.createTrackbar("Grayscale", window, 0, 1, update)       # 0 = off, 1 = on
    cv2.createTrackbar("Threshold On", window, 0, 1, update)    # 0 = off, 1 = on
    cv2.createTrackbar("Threshold", window, 127, 255, update)   # cutoff value
    cv2.createTrackbar("Inverse", window, 0, 1, update)         # 0 = binary, 1 = inverse
    cv2.createTrackbar("Padding", window, 0, 200, update)       # border size in pixels

    update(0)  # show the image once at the start

    # Loop only to handle keys (the callback handles the drawing)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite("edited.jpg", result)
            print("Saved as edited.jpg")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
