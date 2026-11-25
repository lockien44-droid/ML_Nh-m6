"""Read, resize and encode RGB/HSV/LAB images as pixel feature rows."""
from pathlib import Path
import cv2
import numpy as np


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}. Provide --image or edit config.yaml.")
    # imdecode + fromfile hỗ trợ đường dẫn tiếng Việt trên Windows.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path, rgb):
    """Encode RGB as an image file with Unicode-safe paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(path.suffix or ".png", bgr)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def resize_image(image, max_size):
    if not isinstance(max_size, int) or max_size < 1:
        raise ValueError("max_size must be a positive integer.")
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_pixels(array, colorspace="RGB"):
    if colorspace == "RGB":
        encoded, scales = array, np.array([255., 255., 255.])
    elif colorspace == "HSV":
        encoded, scales = cv2.cvtColor(array, cv2.COLOR_RGB2HSV), np.array([179., 255., 255.])
    elif colorspace == "LAB":
        encoded, scales = cv2.cvtColor(array, cv2.COLOR_RGB2LAB), np.array([255., 255., 255.])
    else:
        raise ValueError("colorspace must be RGB, HSV or LAB.")
    return encoded.reshape(-1, 3).astype(float) / scales


def reconstruct_image(features, shape, colorspace="RGB"):
    scales = np.array([179., 255., 255.]) if colorspace == "HSV" else np.array([255., 255., 255.])
    encoded = np.clip(np.rint(features.reshape(shape) * scales), 0, scales).astype("uint8")
    if colorspace == "HSV":
        return cv2.cvtColor(encoded, cv2.COLOR_HSV2RGB)
    if colorspace == "LAB":
        return cv2.cvtColor(encoded, cv2.COLOR_LAB2RGB)
    return encoded


def prepare_image(path, max_size, output=None, colorspace="RGB"):
    original = load_image(path)
    array = resize_image(original, max_size)
    X = to_pixels(array, colorspace)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        save_image(output / "original.png", original)
        save_image(output / "resized.png", array)
        np.savez_compressed(output / "pixels.npz", X=X, shape=np.array(array.shape))
    return X, array, original
