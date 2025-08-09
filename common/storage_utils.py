import concurrent.futures
import os
import shutil
import uuid
from typing import Callable

import requests

from code.semester_project.common.types import DisplayImage, ImageSource


def get_id_from_uri(uri: str) -> str:
    last_slash_index = uri.rfind("/")
    last_dot_index = uri.rfind(".")
    if last_dot_index < last_slash_index:
        return uri[last_slash_index + 1 :]
    if last_slash_index != -1 and last_dot_index != -1:
        return uri[last_slash_index + 1 : last_dot_index]
    print(f"Error extracting ID from {uri}")
    return str(uuid.uuid4())


def get_image_source(image_uri: str) -> ImageSource:
    if image_uri.startswith("http"):
        return ImageSource.URL
    if image_uri.startswith("s3://"):
        return ImageSource.S3
    if image_uri.startswith("gs://"):
        return ImageSource.GCS
    if os.path.exists(image_uri):
        return ImageSource.LOCAL
    raise ValueError(f"Unknown image source for {image_uri}")


def download_image(url: str, local_file_path: str) -> bool:
    """Download image from serving URL

    Args:
    - url (str): http(s):// url of the image
    - local_file_path (str): path on disk where the image is downloaded to
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return False


def copy_file_locally(current_path: str, new_path: str) -> bool:
    try:
        shutil.copy(current_path, new_path)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False


def get_download_fn(image_uri: str) -> Callable[[str, str], bool]:
    source = get_image_source(image_uri)
    match source:
        # case ImageSource.S3:
        #     return download_image_from_s3_url
        case ImageSource.URL:
            return download_image
        # case ImageSource.GCS:
        #     return download_image_from_gcs_url
        case ImageSource.LOCAL:
            return copy_file_locally
        case _:
            raise NotImplementedError(f"No download handler for {source}")


def download_remote_images(
    image_uris: list[str],
    image_ids: list[str] | None = None,
) -> list[DisplayImage]:
    """Download images from cloud storage

    Args:
        - image_uris (list[str])

    Returns:
        - (list[DisplayImage])
    """
    if not image_ids:
        image_ids = [get_id_from_uri(image_uri) for image_uri in image_uris]
    download_handlers = []

    for image_uri in image_uris:
        handler = get_download_fn(image_uri)
        download_handlers.append(handler)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_uri = {
            executor.submit(download, uri, f"{image_id}.jpg"): (
                uri,
                image_id,
            )
            for uri, image_id, download in zip(image_uris, image_ids, download_handlers)
        }

        for future in concurrent.futures.as_completed(future_to_uri):
            uri, image_id = future_to_uri[future]
            try:
                success = future.result()
                if not success:
                    print(f"Download failed on download from {uri}")
            except Exception as e:
                print(f"Error downloading from {uri}: {e}")

    display_images = []
    for image_id, image_uri in zip(image_ids, image_uris):
        display_image = DisplayImage(
            image_id=image_id, path=f"{image_id}.jpg", remote_uri=image_uri
        )
        display_images.append(display_image)
    return display_images


def delete_local_files(local_paths: list[str]):
    for file in local_paths:
        if os.path.exists(file):
            os.remove(file)
