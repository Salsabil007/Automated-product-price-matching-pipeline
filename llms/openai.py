import json

import numpy as np
import requests
from PIL import Image

from code.semester_project.common.constants import OPEN_AI_MODEL_VERSION
from code.semester_project.common.utils import encode_images


class OpenAIModel:
    def __init__(self, verbose: bool = False):
        """Manage OpenAI model functions, such as generating content and getting prices from a grid of price tags"""
        self.api_key = "settings.openai_api_key"
        self.verbose = verbose

    def _convert_numpy_image_to_base64(self, numpy_image: np.ndarray) -> str:
        image = Image.fromarray(numpy_image)
        image_path = "/tmp/image.jpg"
        image.save(image_path)
        return encode_images([image_path])[0]

    def _filter_to_bracketed_text(self, text: str) -> str:
        """Return a slice from `text` that contains everything in between (and including) its outermost square brackets.

        Args:
            text (str): The text to filter.

        Returns:
            str: A slice from `text` of the form "[*]".
        """
        if not text:
            return ""
        i = text.index("[")
        j = -(text[::-1].index("]"))
        return text[i:] if j == 0 else text[i:j]

    def generate_content(self, base64_image: str, prompt: str) -> str:
        """Generate a response using an OpenAI model, given an image and a prompt.

        Args:
            base64_image (str): Path to a base64 image to send to the model.
            prompt (str): The associated prompt for the image to send to the model.

        Returns:
            str: The generated response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": OPEN_AI_MODEL_VERSION,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        ).json()

        if self.verbose:
            usage = response["usage"]
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            print(f"Number of input tokens: {input_tokens}")
            print(f"Number of output tokens: {output_tokens}")
        if "choices" not in response:
            print("choices missing from openai response. response:", response)
            return ""
        return response["choices"][0]["message"]["content"]

    def get_prices(self, grid: np.ndarray, prompt: str) -> list[str | None]:
        """Get prices from an image using the Gemini Pro Vision model.
        Args:
            contents: A tuple containing an Image and a prompt string.
        Returns:
            str: The generated response.
        """
        encoded_grid = self._convert_numpy_image_to_base64(grid)
        response = self.generate_content(encoded_grid, prompt)
        try:
            prices: list[str | None] = json.loads(
                self._filter_to_bracketed_text(response)
            )
        except Exception as e:
            print(f"Invalid response received from OpenAI {OPEN_AI_MODEL_VERSION}. {e}")
            return []
        return prices
