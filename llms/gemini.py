import json
from typing import Literal

import cv2
import numpy as np
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, Image

from code.semester_project.common.constants import GEMINI_PRO_VISION_VERSION


class Gemini:
    """Wrapper class for interacting with Google's Gemini Vision models."""

    def __init__(self, verbose: bool = False):
        """Get a pointer to the Gemini Pro Vision service.

        Args:
            verbose (bool, optional): Whether/not to print verbose output. Right now, this means the # of input/output tokens is printed when the model is called. Defaults to False.
        """
        generation_config = GenerationConfig(
            temperature=0.0,
            candidate_count=1,
            stop_sequences=["\n\n\n"],
        )
        self.model = GenerativeModel(
            GEMINI_PRO_VISION_VERSION, generation_config=generation_config
        )
        self.verbose = verbose

    def _convert_numpy_image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert the given numpy array image representation to bytes.

        Args:
            image (np.ndarray): The image to convert to bytes. (H, W, C) array with RGB channel order.

        Returns:
            bytes: Bytes representation of the image array.
        """
        cv2_image = image[:, :, ::-1]  # CV2 expects BGR channels, not RGB.
        encoded_image = cv2.imencode(".jpg", cv2_image)[1]
        return encoded_image.tobytes()

    def _filter_to_bracketed_text(
        self, text: str, bracket_type: Literal["{", "[", "("] = "["
    ) -> str:
        """Return a slice from `text` that contains everything in between (and including) its outermost `bracket_type` brackets.

        Args:
            text (str): The text to filter.
            bracket_type (Literal["{", "[", "("]): The type of bracket to filter to (curly, square, or round). Defaults to "[".

        Returns:
            str: A slice from `text` in between (and including) its outermost brackets.
        """
        if bracket_type == "{":
            start_bracket, end_bracket = "{", "}"
        elif bracket_type == "[":
            start_bracket, end_bracket = "[", "]"
        else:
            start_bracket, end_bracket = "(", ")"
        i = text.index(start_bracket)
        j = -(text[::-1].index(end_bracket))
        return text[i:] if j == 0 else text[i:j]

    def generate_content(self, image: Image, prompt: str) -> str:
        """Generate a response using the Gemini Pro Vision model, given an image and a prompt

        Args:
            image (Image): The image to send to the model.
            prompt (str): The prompt to send to the model with the image.

        Returns:
            str: The generated response.
        """
        contents = [image, prompt]
        response = self.model.generate_content(contents)
        if self.verbose:
            print(
                f"Number of input tokens: {response.usage_metadata.prompt_token_count}"
            )
            print(
                f"Number of output tokens: {response.usage_metadata.candidates_token_count}"
            )
        return response.text

    def get_prices(self, grid: np.ndarray, prompt: str) -> list[str | None]:
        """Get prices from an image using the Gemini Pro Vision model.

        Args:
            grid (np.ndarray): A grid of price tag images (represented as an array) to send to Gemini.
            prompt (str): The prompt to send for extracting prices.

        Returns:
            str: The generated response.
        """
        encoded_grid = self._convert_numpy_image_to_bytes(grid)
        image = Image.from_bytes(encoded_grid)
        try:
            response = self.generate_content(image, prompt)
        except ValueError as e:
            print(f"Error extracting prices: {e}")
            return []

        try:
            prices: list[str | None] = json.loads(
                self._filter_to_bracketed_text(response)
            )
        except Exception as e:
            print(f"Invalid response received from Gemini. {e}")
            return []
        return prices
