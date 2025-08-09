from typing import Iterator, Sequence, TypeVar
import base64
import cv2

T = TypeVar("T")


def chunker(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def find(filter_func, items: list[T]) -> T | None:
    res = list(filter(filter_func, items))
    return res[0] if len(res) > 0 else None


def encode_images(image_paths: list[str]) -> list[str]:
    """
    Encodes images to base64

    Args:
        image_paths: list of image paths

    Returns:
        list of base64 encoded images
    """
    encoded_images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        jpg_img = cv2.imencode(".jpg", img)
        b64_string = base64.b64encode(jpg_img[1]).decode("utf-8")
        encoded_images.append(b64_string)
    return encoded_images
