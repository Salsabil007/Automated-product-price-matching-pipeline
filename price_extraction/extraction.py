from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL.Image import Image as PILImage
from ultralytics import YOLO
from ultralytics.engine.results import Results

from code.semester_project.common.constants import PRICE_TAG_LABEL
from code.semester_project.common.utils import chunker
from code.semester_project.llms.gemini import Gemini


def format_price(price: str) -> str | None:
    price = price.lower()
    if "/" in price or "for" in price:
        if "/" in price:
            split = "/"
        else:
            split = "for"
        split_values = price.replace("$", "").split(split)
        if len(split_values) == 2 and split_values[0] and split_values[1]:
            quantity, total_price = split_values
            total_price = total_price.strip()
            if " " in total_price:
                total_price = total_price.replace(" ", ".")
            if "." not in total_price:
                if len(total_price) > 2:
                    total_price = f"{total_price[:-2]}.{total_price[-2:]}"
                else:
                    total_price = f"{total_price}.00"
            price = f"{quantity.strip()} for ${total_price}"
        else:
            return None
    if price.endswith("¢"):
        price = f"0.{price[:-1]}"
    if price.startswith("$"):
        price = price[1:]
    if len(price) == 2 and price.isdigit() and float(price) >= 50:
        price = f"0.{price}"
    if len(price) == 3 and price.isdigit():
        price = f"{price[0]}.{price[1:]}"
    return price


def is_valid_price(price: str | None) -> bool:
    """Check if the given string is a valid price (e.g. exists and fits into expected formats).
    Args:
        price (str): The string to check.
    Returns:
        bool: True if the string is a valid price, False otherwise.
    """
    if price is None:
        return False
    if "buy" in price.lower():
        return True
    price = price.replace("$", "").strip()
    if "for" in price:
        if len(price.split()) != 3:
            return False
        if "$" in price:
            quantity, total_price = price.split(" for $")
        else:
            quantity, total_price = price.split(" for ")
        if not quantity.isdigit() or not total_price.replace(".", "").isdigit():
            return False
        if (
            total_price.replace(".", "").isdigit()
            and quantity.isdigit()
            and float(total_price) / int(quantity) < 0.5
        ):
            return False
    else:
        if price.count(".") > 1 or not price.replace(".", "").isdigit():
            return False
        try:
            if float(price) < 0.5:
                return False
        except ValueError:
            return False
    if "." not in price:
        return False
    return True


@dataclass
class DisplayImagePrices:
    """Representation of all unique prices found in a display image (along with the corresponding bounding box)."""

    prices: list[str | None]
    bboxes: list[np.ndarray]


class PriceExtractor:
    """An object to recognize and extract all prices visible in the given display images.

    Args:
        meta_bucket_path (str): GCS bucket path to meta model (should be .pt).
        verbose (bool, optional): Enable/disable verbose output. Currently, this just enables/disabled printing the input/output tokens for each Gemini call.
    """

    def __init__(
        self,
        meta_detector_bucket_path: str | None = None,
        meta_detector: YOLO | None = None,
        verbose: bool = False,
    ):
        if meta_detector is not None:
            self._meta_detector = meta_detector
        elif meta_detector_bucket_path:
            self._meta_detector = self._download_and_load_meta_detector(
                meta_detector_bucket_path
            )
        else:
            raise ValueError(
                "Either a meta detector or a meta detector bucket path must be provided."
            )
        # TODO: Setup Gen models? Decide if we want these
        # if settings.gen_model == "gemini":
        self._gen_model = Gemini(verbose=verbose)
        # elif settings.gen_model == "openai":
        #     self._gen_model = OpenAIModel(verbose=verbose)
        # else:
        #     raise Exception("settings.gen_model must be 'gemini' or 'openai'")
        self._prompt = self._load_prompt()

    def extract_prices_from_display_image(
        self, image: PILImage, bboxes: list[np.ndarray] | None = None
    ) -> DisplayImagePrices:
        """Extracts pricing data from given price tag bounding boxes

        Args:
            image (PILImage): Display image to extract prices from.
            bboxes (list[np.ndarray]): xyxy bounding box coordinates for the price tags.
                If not provided, we extract them

        Returns:
            DisplayImagePrices: The visible prices in the display image, along with the associated bounding boxes.
        """

        if not bboxes:
            bboxes = self._get_price_tag_bboxes(image)

        if len(bboxes) == 0:
            return DisplayImagePrices(prices=[], bboxes=[])

        is_relative = bboxes[0].max() <= 1
        if is_relative:
            width, height = image.size
            crops = [
                np.array(image)[
                    int(min_y * height) : int(max_y * height),
                    int(min_x * width) : int(max_x * width),
                ]
                for (min_x, min_y, max_x, max_y) in bboxes
            ]
        else:
            crops = [
                np.array(image)[int(min_y) : int(max_y), int(min_x) : int(max_x)]
                for (min_x, min_y, max_x, max_y) in bboxes
            ]

        prices = self._extract_price_text_from_crops(crops)
        return DisplayImagePrices(prices=prices, bboxes=bboxes)

    def _extract_price_text_from_crops(
        self, crops: list[np.ndarray]
    ) -> list[str | None]:
        """Extracts visible price text from cropped images using the generative model (Gemini/GPT)

        Args:
            crops (list[np.ndarray]): cropped ndarray images for price tags

        Returns:
            prices (list[str]): List of extracted prices. Indices match up with crops
        """

        all_prices = []
        i = 0
        padding = 200
        num_columns = 5
        # We handle crops in chunks of 5 so the grid doesn't get too unwieldy
        for crops_chunk in chunker(crops, size=num_columns):
            # The prompt makes the model assume more than one price tag, so weird
            # things happen if the grid only has one price tag in it. We artificially
            # pad in this case, then take the first price only.
            num_grid_rows = 1
            num_grid_columns = min(len(crops_chunk), num_columns)
            duplicated = False
            if len(crops_chunk) == 1:
                num_grid_columns = 3
                grid = self._arrange_price_tag_crops_in_grid(
                    crops_chunk * 3,
                    grid_dims=(num_grid_rows, num_grid_columns),
                    padding=padding,
                )
                duplicated = True
            else:
                grid = self._arrange_price_tag_crops_in_grid(
                    crops_chunk,
                    grid_dims=(num_grid_rows, num_grid_columns),
                    padding=padding,
                )
            i += 1
            prices = self._gen_model.get_prices(grid, prompt=self._prompt)[
                : len(crops_chunk)
            ]
            prices = [price if price != "null" else None for price in prices]

            # If extracted prices doesn't match the number of images, default all to null
            total_num_prices = num_grid_rows * num_grid_columns
            if len(prices) != total_num_prices and not (
                duplicated and len(prices) == 1
            ):
                prices = [None for _ in range(total_num_prices)]
                all_prices.extend(prices)
            else:
                for i, price in enumerate(prices):
                    if price:
                        price = format_price(price)
                        if is_valid_price(price):
                            prices[i] = price
                        else:
                            prices[i] = None
                all_prices.extend(prices)
        return all_prices

    def _download_and_load_meta_detector(self, meta_bucket_path: str) -> YOLO:
        """Download the meta detector at the given bucket path and return its YOLO instance.

        Args:
            meta_bucket_path (str): GCS bucket path to the meta detector.

        Returns:
            YOLO: The meta detector.
        """
        local_meta_path = "meta_detector.pt"
        download_blob(meta_bucket_path, local_meta_path, skip_if_present=True)
        return YOLO(local_meta_path)

    def _load_prompt(self) -> str:
        """Load the prompt used to get prices on a display from the generative model.

        Returns:
            str: The prompt.
        """
        prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "prompts"
            / "price_extraction_prompt.txt"
        )
        with open(prompt_path, "r") as f:
            prompt = f.read()
        return prompt

    def _get_price_tag_bboxes(self, image: PILImage) -> list[np.ndarray]:
        """From the given display image, detect price tags and return the resultant bounding boxes.

        Args:
            image (PILImage): The display image containing the price tags.

        Returns:
            list[np.ndarray]: List of price tag bounding boxes (in xyxy format).
        """
        results: list[Results] = self._meta_detector.predict(image)
        bboxes = [
            np.round(box.xyxy.detach().numpy().flatten()).astype(int)
            for box in results[0].boxes
            if self._meta_detector.names[box.cls.item()] == PRICE_TAG_LABEL
        ]
        return bboxes

    def _draw_outlines_around_tags(
        self,
        grid_image: np.ndarray,
        x_offset: int,
        y_offset: int,
        crop_width: int,
        crop_height: int,
    ) -> np.ndarray:
        line_thickness = 5
        # Top red line
        grid_image[
            y_offset : y_offset + line_thickness, x_offset : x_offset + crop_width
        ] = [
            255,
            0,
            0,
        ]
        # Bottom red line
        grid_image[
            y_offset + crop_height - line_thickness : y_offset + crop_height,
            x_offset : x_offset + crop_width,
        ] = [255, 0, 0]
        # Left red line
        grid_image[
            y_offset : y_offset + crop_height, x_offset : x_offset + line_thickness
        ] = [
            255,
            0,
            0,
        ]
        # Right red line
        grid_image[
            y_offset : y_offset + crop_height,
            x_offset + crop_width - line_thickness : x_offset + crop_width,
        ] = [255, 0, 0]
        return grid_image

    def _arrange_price_tag_crops_in_grid(
        self,
        crops: Iterable[np.ndarray],
        grid_dims: tuple[int, int] = (1, 5),
        padding: int = 200,
    ) -> np.ndarray:
        """Given a collection of price tag crops, arrange them on a blank grid with padding.

        Args:
            crops (list[np.ndarray]): The price tag crops to arrange in a grid.
            grid_dims (tuple[int, int], optional): Number of rows/columns in the grid. Defaults to (1, 5), which has produced good Gemini results.
            padding (int, optional): The amount of padding pixels to put between crops in the grid. Defaults to 200.

        Returns:
            np.ndarray: Array representation of the price tag grid, with dimensions (H, W, C). Channel order is RGB.
        """
        num_rows, num_cols = grid_dims
        max_height = max(crop.shape[0] for crop in crops)
        max_width = max(crop.shape[1] for crop in crops)
        grid_width = num_cols * (max_width + padding) - padding
        grid_height = num_rows * (max_height + padding) - padding

        grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

        for i, crop in enumerate(crops):
            row = i // num_cols
            col = i % num_cols
            crop_height, crop_width = crop.shape[:2]
            x_offset = col * (max_width + padding)
            y_offset = row * (max_height + padding)
            grid_image[
                y_offset : y_offset + crop_height, x_offset : x_offset + crop_width
            ] = crop
            grid_image = self._draw_outlines_around_tags(
                grid_image, x_offset, y_offset, crop_width, crop_height
            )

        return grid_image
