from typing import Any

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from code.semester_project.common.storage_utils import (
    download_remote_images,
    delete_local_files,
)
from code.semester_project.common.types import (
    ImageProductPriceAttributions,
    ImageProductPriceDetections,
    Price,
    ProductPrice,
)

from code.semester_project.price_extraction.attribution import (
    NearestBelowWithinPkgGroupAttribution,
)
from code.semester_project.price_extraction.extraction import PriceExtractor


def get_product_packaging_groups(product_info: dict[str, dict[str, Any]]):
    upc_to_group_name = {}
    group_to_upcs = {}
    for upc, info in product_info.items():
        pkg_info = info["packaging"]
        pkg_str = f"{pkg_info['quantity']} {pkg_info['size']}{pkg_info['unit']} {pkg_info['container']}"
        pkg_manu_str = f"{info['manufacturer']} {pkg_str}"
        if pkg_manu_str not in group_to_upcs:
            group_to_upcs[pkg_manu_str] = []
        group_to_upcs[pkg_manu_str].append(upc)
        upc_to_group_name[upc] = pkg_manu_str
    upc_to_group_upcs = {}
    for upc, group_name in upc_to_group_name.items():
        upc_to_group_upcs[upc] = group_to_upcs[group_name]
    return upc_to_group_upcs


def update_prices_based_on_packaging_group_voting(
    upc_to_price: dict[str, Price],
    upc_to_group_upcs: dict[str, list[str]],
) -> dict[str, Price]:
    """Update prices based on voting within packaging group_upc, taking the most common price in each group

    Args:
        upc_to_price: dict mapping upc to attributed price
        upc_to_group_upcs: dict mapping upc to the upcs in its same packaging group

    Returns:
        dict mapping upc to the most common price in its packaging group
    """
    upc_to_voted_price = {}
    for upc, group_upcs in upc_to_group_upcs.items():
        if upc not in upc_to_price:
            continue
        group_prices = [
            upc_to_price[group_upc]
            for group_upc in group_upcs
            if (group_upc in upc_to_price and upc_to_price[group_upc])
        ]
        if not group_prices:
            upc_to_voted_price[upc] = upc_to_price[upc]
        else:
            most_common_price = max(set(group_prices), key=group_prices.count)
            upc_to_voted_price[upc] = most_common_price
    return upc_to_voted_price


def check_plausible_prices(
    upc_to_price: dict[str, Price],
    upc_to_product_info: dict[str, Any],
    plausible_prices: dict[str, Any],
) -> dict[str, Price]:
    """Remove prices that are not within the plausible price range

    Args:
        upc_to_price: dict mapping upc to attributed price
        upc_to_product_info: dict mapping upc to product info like category and packaging
        plausible_prices: dict mapping category to plausible price range with min and max

    Returns:
        dict mapping upc to price, with prices outside the plausible price range set to None
    """
    for upc, price in upc_to_price.items():
        if price is None:
            continue
        category = upc_to_product_info[upc]["category"]
        if category in plausible_prices:
            category_prices = plausible_prices.get(category, {})
            if isinstance(price, str) and "for" in price:
                quantity, total_price = price.split(" for $")
                unit_price = float(total_price) / int(quantity)
            else:
                unit_price = float(price)
            pkg_info = upc_to_product_info[upc]["packaging"]
            pkg_str = f"{pkg_info['quantity']} {pkg_info['size']}{pkg_info['unit']}"
            pkg_prices = category_prices.get(pkg_str, {})
            min_ = pkg_prices.get("min", None)
            max_ = pkg_prices.get("max", None)
            if min_ is not None and max_ is not None:
                if unit_price < min_ or unit_price > max_:
                    upc_to_price[upc] = None
    return upc_to_price


def map_attributions_to_product_prices(
    attributions: dict[str, Price],
) -> list[ProductPrice]:
    """Map Price attributions to ProductPrice objects

    Args:
        attributions: dict mapping upc to price

    Returns:
        list of ProductPrice objects
    """
    product_prices = []
    for label, price in attributions.items():
        if not price or label == "Unidentified Product":
            continue
        product_prices.append(ProductPrice(label=label, prices=[price]))
    return product_prices


def get_products_prices_from_image(
    meta_detector: YOLO,
    image_data: ImageProductPriceDetections,
    plausible_prices: dict,
    upc_to_product_info: dict[str, Any],
) -> ImageProductPriceAttributions:
    image_path = image_data.image_path
    image = Image.open(image_path)

    bounding_boxes = [bb.to_ndarray() for bb in image_data.price_tag_bounding_boxes]

    price_tag_text_predictions = PriceExtractor(
        meta_detector=meta_detector
    ).extract_prices_from_display_image(image=image, bboxes=bounding_boxes)

    price_tags = []
    for idx, price_tag in enumerate(image_data.price_tag_bounding_boxes):
        box = price_tag.model_dump()
        box["price_tag_text"] = price_tag_text_predictions.prices[idx]
        price_tags.append(box)

    data_dict = image_data.model_dump()

    upc_to_group_upcs = get_product_packaging_groups(upc_to_product_info)

    attribution = NearestBelowWithinPkgGroupAttribution(
        data_dict["product_bounding_boxes"],
        price_tags,
        upc_to_group_upcs=upc_to_group_upcs,
    )
    label_price_attributions = attribution.run()

    label_price_attributions = update_prices_based_on_packaging_group_voting(
        label_price_attributions, upc_to_group_upcs
    )
    label_price_attributions = check_plausible_prices(
        label_price_attributions, upc_to_product_info, plausible_prices
    )

    product_prices = map_attributions_to_product_prices(label_price_attributions)
    return ImageProductPriceAttributions(image_path=image_path, prices=product_prices)


def get_product_prices_from_images(
    images: list[ImageProductPriceDetections],
) -> list[ImageProductPriceAttributions]:
    image_uris = [i.image_path for i in images]
    display_images = download_remote_images(image_uris)
    # upcs = [box.label for image in images for box in image.product_bounding_boxes]
    # TODO will need to get "plausible_prices" and "product_info" some other way? Talk to Andrew
    # product_info = client.get_product_info(upcs)
    # plausible_prices = ultra_client.get_plausible_prices()
    meta_detector = YOLO(
        # TODO: setup model files
    )

    all_image_prices: list[ImageProductPriceAttributions] = []
    for data in tqdm(images):
        all_image_prices.append(
            get_products_prices_from_image(
                meta_detector=meta_detector,
                image_data=data,
                plausible_prices={},
                upc_to_product_info={},
            )
        )

    delete_local_files([image.path for image in display_images])

    return all_image_prices
