import matplotlib.pyplot as plt

# Sharda , A. (2021, January 12). Understanding image contrast algorithms. 
# Medium. https://aryamansharda.medium.com/understanding-image-contrast-algorithms-8636723a0f05 

import cv2
import numpy as np
import time

# --- operations stay as standalone functions (fully modular)  --------------------
def apply_brightness(img, value):
    # (pixel + value) goes beyond 255, or below 0
    # img pixels are uint8, so we need to convert to int16 to avoid overflow, then add value
    # result = img.astype(np.int16) + value
    result = img.astype(np.int16) * (1 + (value/100))
    return np.clip(result, 0, 255).astype(np.uint8)
# To do: improve the brightness function
# "The previous code is overly simplistic. A proper implementation would map the RGB color space to HSL. 
# Then, it would modify the luminance values and then convert the image back to the RGB color space. 
# However, modifying the RGB values in this way still illustrates the point without increasing the article’s scope."
# (Sharda , A. (2021, January 12))

# to increase contrast, we need to increase the distance between the maximum and minimum pixel intensities
# pixel(50): 1.5*(50-128)+128 = 11 (darker)
# pixel(128): stays 128
# pixel(200): 1.5*(200-128)+128 = 236 (brighter)
def apply_contrast(img, factor):
    result = factor * (img.astype(np.float32) - 128) + 128
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # back to 3 channels


def apply_threshold(img, value, inverse):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, value, 255, mode)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

# ratio = width/height
# width = height * ratio
# height = width/ratio
def apply_padding(img: np.ndarray, size: int, border_type: str = "constant", ratio:float=None) -> np.ndarray:
    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    image_h, image_w = img.shape[:2]
    border_t = border_map.get(border_type)

    if ratio is None:
        return cv2.copyMakeBorder(img, size, size, size, size, border_t)
    else:
        # find the image size with padding applied equally (x2, 2 sides)
        padded_w = image_w + 2 * size
        padded_h = image_h + 2 * size

        # add padding to meet ratio (determine final width, height with padding)
        # if the ratio only requires increasing 1 dimension, it chooses only that dimension to pad
        final_w = max(padded_w, int(round(padded_h * ratio)))
        final_h = max(padded_h, int(round(padded_w / ratio)))

        # determine padding to add to each side
        padding_w = size + (final_w - padded_w)//2
        padding_h = size + (final_h - padded_h)//2
        # cv2.putText(result, f"Border: {border_type}", (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return cv2.copyMakeBorder(img, padding_h,padding_h, padding_w,padding_w, border_t)

def apply_blend(img, other_path, alpha):
    other = cv2.imread(other_path)
    if other is None:
        raise FileNotFoundError(f"Could not load image: {other_path}")
    other = cv2.resize(other, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, alpha, other, 1 - alpha, 0)

def show_comparison(original, preview):
    # OpenCV=BGR, matplotlib=RGB
    o = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    p = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(o); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(p); ax[1].set_title("Preview"); ax[1].axis("off")
    plt.show()

class PhotoEditor:
    def __init__(self, path):
        self.original = cv2.imread(path)
        self.result = self.original.copy()
        self.window = "Photo Editor"
        # UI update timing
        self.needs_update = True
        self.last_change_time = 0.0
        self.debounce_ms = 100
        # state variables
        self.base_original = self.original.copy()  # for matplotlib comparison
        self.undo_stack = []
        self.history = []
        self.preview_settings = {}


    def now_ms(self):
        return time.monotonic() * 1000

    # records that something changed
    def on_change(self, x):
        self.needs_update = True
        self.last_change_time = self.now_ms()
    
    def create_window(self):
        cv2.namedWindow(self.window)
        cv2.createTrackbar("Brightness", self.window, 100, 200, self.on_change)
        cv2.createTrackbar("Contrast", self.window, 10, 30, self.on_change)
        cv2.createTrackbar("Grayscale", self.window, 0, 1, self.on_change)
        cv2.createTrackbar("Threshold On", self.window, 0, 1, self.on_change)
        cv2.createTrackbar("Threshold", self.window, 127, 255, self.on_change)
        cv2.createTrackbar("Inverse", self.window, 0, 1, self.on_change)
        cv2.createTrackbar("Padding", self.window, 0, 200, self.on_change)
        # 0=constant, 1=reflect, 2=replicate
        cv2.createTrackbar("Border", self.window, 0, 2, self.on_change)
        # 0=rectangle, 1=square, 2=4:5, 3=16:9
        cv2.createTrackbar("Proportion", self.window, 0, 3, self.on_change)

        
    # --- Callback: runs every time ANY slider moves ----------------------------
    def update(self):
        global result

        # Read the current slider positions
        brightness = cv2.getTrackbarPos("Brightness", self.window) - 100
        contrast = cv2.getTrackbarPos("Contrast", self.window) / 10
        grayscale = cv2.getTrackbarPos("Grayscale", self.window)
        thresh_on = cv2.getTrackbarPos("Threshold On", self.window)
        thresh_val = cv2.getTrackbarPos("Threshold", self.window)
        inverse = cv2.getTrackbarPos("Inverse", self.window)
        padding = cv2.getTrackbarPos("Padding", self.window)
        border = cv2.getTrackbarPos("Border", self.window)
        border_type = ["constant", "reflect", "replicate"][border]
        proportion = cv2.getTrackbarPos("Proportion", self.window)
        ratio = {0: None, 1: 1.0, 2: 4/5, 3: 16/9}[proportion]
        proportion_label = ["rectangle", "square", "4:5", "16:9"][proportion]

        self.preview_settings = {
            "brightness": brightness,
            "contrast": contrast,
            "grayscale": bool(grayscale),
            "thresh_on": bool(thresh_on),
            "thresh_val": thresh_val,
            "inverse": bool(inverse),
            "padding": padding,
            "border_type": border_type,
            "proportion": proportion,
            "proportion_label": proportion_label,
            "ratio": ratio,
        }

        # Apply the operations one after another
        result = self.original.copy()
        result = apply_brightness(result, brightness)
        result = apply_contrast(result, contrast)
        if grayscale:
            result = apply_grayscale(result)
        if thresh_on:
            result = apply_threshold(result, thresh_val, inverse)
        if padding:
            result = apply_padding(result, padding, border_type=border_type, ratio=ratio)
            if ratio is not None:
                h, w = result.shape[:2]
                print(f"got {w/h:.4f}, want {ratio:.4f}")
        self.result = result
        cv2.imshow(self.window, result)

    def build_history_entry(self):
        s = self.preview_settings
        if not s:
            return "no changes"

        parts = []
        brightness = s.get("brightness", 0)
        if brightness != 0:
            parts.append(f"brightness {brightness:+d}")

        contrast = s.get("contrast", 1.0)
        # greater than 10^-3 (not 1/e^3)
        if abs(contrast - 1.0) > 1e-3:
            parts.append(f"contrast x{contrast:.2f}")

        if s.get("grayscale"):
            parts.append("grayscale")

        if s.get("thresh_on"):
            mode = "inverse" if s.get("inverse") else "binary"
            parts.append(f"threshold {mode} {s.get('thresh_val', 0)}")

        padding = s.get("padding", 0)
        ratio = s.get("ratio", None)
        if padding > 0 or ratio is not None:
            border = s.get("border_type", "constant")
            prop_label = s.get("proportion_label", "rectangle")
            parts.append(f"padding {padding}px {border} {prop_label}")

        if not parts:
            return "no changes"

        return "; ".join(parts)

    def handle_key(self, key):
        # APPLY current preview
        if key == ord("a"):  # a
            self.undo_stack.append(self.original.copy())
            self.original = self.result.copy()
            self.history.append(self.build_history_entry())
            show_comparison(self.base_original, self.original)

            # reset sliders so next edits don't stack
            cv2.setTrackbarPos("Brightness", self.window, 100)
            cv2.setTrackbarPos("Contrast", self.window, 10)
            cv2.setTrackbarPos("Grayscale", self.window, 0)
            cv2.setTrackbarPos("Threshold On", self.window, 0)
            cv2.setTrackbarPos("Inverse", self.window, 0)
            cv2.setTrackbarPos("Padding", self.window, 0)
            cv2.setTrackbarPos("Border", self.window, 0)
            cv2.setTrackbarPos("Proportion", self.window, 0)
            self.needs_update = True

        # UNDO
        elif key == ord("u"):
            if self.undo_stack:
                self.original = self.undo_stack.pop()
                if self.history:
                    self.history.pop()
                self.needs_update = True

        # HISTORY
        elif key == ord("h"):
            print("History:")
            if not self.history:
                print("  (empty)")
            else:
                for i, step in enumerate(self.history, 1):
                    print(f"  {i}. {step}")

        # BLEND
        elif key == ord("b"):
            try:
                path = input("Second image path: ").strip().strip('"')
                alpha = float(input("Alpha 0..1: ") or "0.5")
                blended = apply_blend(self.original, path, alpha)
            except (FileNotFoundError, ValueError, cv2.error, OSError) as e:
                print(f"Blend failed: {e}")
            else:
                self.undo_stack.append(self.original.copy())
                self.original = blended
                self.history.append(f"blend {path}, alpha {alpha:.2f}")
                show_comparison(self.base_original, self.original)
                self.needs_update = True

        # QUIT + SAVE
        elif key == ord("q"):
            print("History before exit:")
            for i, step in enumerate(self.history, 1):
                print(f"  {i}. {step}")
            save = input("Save final image? (y/n): ").strip().lower()
            if save == "y":
                name = input("Filename: ").strip() or "edited.jpg"
                cv2.imwrite(name, self.original)
                print(f"Saved as {name}")
            return True  # signal quit

        return False
        
# --- Main program ----------------------------------------------------------
def main():

    editor = PhotoEditor("cat.jpg")
    # create instance
    editor.create_window()
    editor.update()

    while True:
        if editor.needs_update and (editor.now_ms() - editor.last_change_time) >= editor.debounce_ms:
            editor.update()
            editor.needs_update = False

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # key was pressed
            quit_now = editor.handle_key(key)
            if quit_now:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
