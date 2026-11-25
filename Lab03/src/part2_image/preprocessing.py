"""Read, resize and encode RGB/HSV/LAB images as pixel feature rows."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}. Provide --image or edit config.yaml.")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resize_image(image, max_size):
    if not isinstance(max_size, int) or max_size < 1:
        raise ValueError("max_size must be a positive integer.")
    working = image.copy()
    working.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(working)


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
        original.save(output / "original.png")
        Image.fromarray(array).save(output / "resized.png")
        np.savez_compressed(output / "pixels.npz", X=X, shape=np.array(array.shape))
    return X, array, original
