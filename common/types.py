from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel

Price = str | float | None


class BoundingBox(BaseModel):
    label: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __repr__(self):
        return f"{self.label}: [{self.min_x},{self.min_y},{self.max_x},{self.max_y}]"

    def to_ndarray(self) -> np.ndarray:
        return np.array([self.min_x, self.min_y, self.max_x, self.max_y])

    def to_dict(self):
        return {
            "label": self.label,
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }

    @classmethod
    def from_dict(cls, box_json: dict) -> "BoundingBox":
        if "min_x" not in box_json and "x1" in box_json:
            raise ValueError(
                f"box_json must have either min_x/min_y/max_x/max_y or x1/y1/x2/y2 format ({box_json})"
            )
        return BoundingBox(
            label=box_json["label"],
            min_x=box_json.get("min_x", box_json.get("x1", None)),
            min_y=box_json.get("min_y", box_json.get("y1", None)),
            max_x=box_json.get("max_x", box_json.get("x2", None)),
            max_y=box_json.get("max_y", box_json.get("y2", None)),
        )

    @classmethod
    def create_boxes_from_yolo_output(cls, yolo_output):
        boxes = []
        for pred in yolo_output:
            label = pred["name"]
            box = pred["box"]
            min_x = box["x1"]
            min_y = box["y1"]
            max_x = box["x2"]
            max_y = box["y2"]
            boxes.append(
                BoundingBox(
                    label=label,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                )
            )
        return boxes


class ImageProductPriceDetections(BaseModel):
    image_path: str
    product_bounding_boxes: list[BoundingBox]
    price_tag_bounding_boxes: list[BoundingBox]


class PriceAttributionRequest(BaseModel):
    images: list[ImageProductPriceDetections]


class ProductPrice(BaseModel):
    label: str
    prices: list[Price]


class ImageProductPriceAttributions(BaseModel):
    image_path: str
    prices: list[ProductPrice]


class PricingInfo(BaseModel):
    price: Price = None
    sale_price: Price = None
    multi_price: Price = None
    unit_price: Price = None

    @classmethod
    def is_single_price(cls, price: Price) -> bool:
        if price is None:
            return False
        try:
            float(price)
            return True
        except ValueError:
            return False

    @classmethod
    def from_list(cls, prices: list[Price]) -> "PricingInfo":
        """
        Convert a list of prices to PricingInfo
        Args:
            prices (list[str]): The list of prices, mix of single floats and multi price strings
        Returns:
            PricingInfo: The PricingInfo object
        """
        price_dict: dict[str, Price] = {
            "price": None,
            "sale_price": None,
            "multi_price": None,
            "unit_price": None,
        }
        single_prices = [
            price for price in prices if PricingInfo.is_single_price(price)
        ]
        multi_prices = [price for price in prices if price not in single_prices]

        if single_prices:
            for price in single_prices:
                if price is None:
                    continue
                price = float(price)
                if not price_dict["price"]:
                    price_dict["price"] = price
                if price_dict["price"]:
                    if price < float(price_dict["price"]):
                        price_dict["sale_price"] = price
                    elif price > float(price_dict["price"]):
                        price_dict["sale_price"] = price_dict["price"]
                        price_dict["price"] = price
                elif "sale_price" in price_dict and price_dict["sale_price"]:
                    if price < float(price_dict["sale_price"]):
                        price_dict["price"] = price_dict["sale_price"]
                        price_dict["sale_price"] = price
                    elif price > float(price_dict["sale_price"]):
                        price_dict["price"] = price
        if multi_prices:
            # Only handling one multi_price for now
            price_dict["multi_price"] = multi_prices[0]

        pricing_info = cls(**price_dict)

        return pricing_info

    def to_dict(self):
        output = {}
        for key in ["price", "sale_price", "multi_price", "unit_price"]:
            value = getattr(self, key)
            if value is not None:
                output[key] = value
        return output


class ProductPricingInfo(BaseModel):
    upc: str
    pricing: PricingInfo

    @classmethod
    def from_prices(cls, upc: str, prices: list[Price]) -> "ProductPricingInfo":
        return cls(upc=upc, pricing=PricingInfo.from_list(prices))

    def to_dict(self):
        return {
            "upc": self.upc,
            "pricing": self.pricing.to_dict(),
        }


class ImageProductPricingInfo(BaseModel):
    image_path: str
    pricing: list[ProductPricingInfo]


class ImagePricingInfo(BaseModel):
    image_id: str
    pricing: list[ProductPricingInfo]


@dataclass
class DisplayImage:
    image_id: str
    path: str
    remote_uri: str


class ImageSource(Enum):
    S3 = "s3"
    GCS = "gcs"
    URL = "url"
    LOCAL = "local"
    UNKNOWN = "unknown"
