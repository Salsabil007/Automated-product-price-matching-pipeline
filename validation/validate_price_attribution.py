import argparse
import json
import os

from dotenv import load_dotenv

from code.semester_project.common.types import BoundingBox, ImageProductPriceDetections
from code.semester_project.validation.image_price_attribution import (
    get_product_prices_from_images,
)

load_dotenv()


def validate(dataset_dir: str):  # noqa: C901
    if dataset_dir.endswith("/"):
        dataset_dir = dataset_dir[:-1]
    dataset_file = os.path.join(dataset_dir, "dataset.json")
    bboxes_file = os.path.join(dataset_dir, "bounding_boxes.json")
    with open(os.path.join(dataset_dir, "upc_to_name.json")) as f:
        upcs_and_names = json.load(f)
    upc_to_name = {item["upc_a"]: item["name"] for item in upcs_and_names}

    with open(bboxes_file, "r") as f:
        bboxes = json.load(f)

    images = []
    for image_boxes in bboxes:
        image_path = os.path.join(dataset_dir, image_boxes["img_path"])
        product_bboxes = [
            BoundingBox.from_dict(bbox)
            for bbox in image_boxes["product_bounding_boxes"]
        ]
        price_tag_bboxes = [
            BoundingBox.from_dict(bbox)
            for bbox in image_boxes["price_tag_bounding_boxes"]
        ]
        images.append(
            ImageProductPriceDetections(
                image_path=image_path,
                product_bounding_boxes=product_bboxes,
                price_tag_bounding_boxes=price_tag_bboxes,
            )
        )

    gt_prices = {}
    img_path_to_gt_count = {}
    total_gt_labels = 0
    total_gt_prices = 0
    img_path_to_gt_product_count = {}
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
        for image in dataset:
            image_path = image["img_path"]
            filename = image_path.split("/")[-1]
            img_path_to_gt_count[filename] = len(image["products"])
            img_path_to_gt_product_count[filename] = len(
                {prod["upc"] for prod in image["products"]}
            )
            products = image["products"]
            total_gt_labels += len(products)
            for product in products:
                upc = product["upc"]
                all_prices = product["price_info"]
                multi_price = all_prices["multi_price"]
                prices = [
                    str(price)
                    for price in [
                        all_prices["price"],
                        all_prices["sale_price"],
                        multi_price,
                    ]
                    if price
                ]
                total_gt_prices += len(prices)
                if filename not in gt_prices:
                    gt_prices[filename] = {}
                if upc not in gt_prices[filename]:
                    gt_prices[filename][upc] = {
                        "prices": prices,
                        "unit_prices": [all_prices["unit_price"]],
                    }
                else:
                    gt_prices[filename][upc]["prices"].extend(prices)
                    gt_prices[filename][upc]["unit_prices"].append(
                        all_prices["unit_price"]
                    )
            gt_prices[filename]["all"] = {
                price
                for upc in gt_prices[filename]
                for price in gt_prices[filename][upc]["prices"]
            }

    all_image_product_prices = get_product_prices_from_images(images)
    total_prices_predicted = 0
    total_predicted_prices_correct = 0
    total_prices_correct = 0
    total_price_error_from_gen_model = 0
    total_price_error_from_attribution = 0

    for image_product_prices in all_image_product_prices:
        print("\n\nimage path:", image_product_prices.image_path)
        image_path = image_product_prices.image_path
        filename = image_path.split("/")[-1]
        product_prices = image_product_prices.prices
        valid_predictions = []
        for prices in product_prices:
            valid_predictions.extend(
                [p for p in prices.prices if (p is not None and p != "null" and p)]
            )
        image_predicted = len(valid_predictions)
        image_correct_predicted = 0
        image_errors_from_gen_model = 0
        image_errors_from_attribution = 0
        attributed = []
        missed = []

        for product_price in product_prices:
            label = product_price.label
            pred_prices = product_price.prices
            valid_pred_prices = [
                p for p in pred_prices if (p is not None and p != "null" and p)
            ]
            image_gt_prices = gt_prices[filename][label]["prices"]
            for price in valid_pred_prices:
                instance = {
                    "image_path": filename,
                    "label": label,
                    "pred": price,
                    "gt": image_gt_prices,
                    "name": upc_to_name.get(label, "Unknown"),
                }
                if price in image_gt_prices:
                    image_correct_predicted += 1
                    attributed.append(instance)
                else:
                    if price in gt_prices[filename]["all"]:
                        total_price_error_from_attribution += 1
                        image_errors_from_attribution += 1
                    else:
                        total_price_error_from_gen_model += 1
                        image_errors_from_gen_model += 1
                    missed.append(instance)
        labels_not_predicted = [
            (
                label,
                gt_prices[filename][label]["prices"],
                upc_to_name.get(label, "Unknown"),
            )
            for label in gt_prices[filename]
            if label != "all" and label not in [p.label for p in product_prices]
        ]
        image_correct_total = image_correct_predicted
        for _, gt_price_list, _ in labels_not_predicted:
            if not gt_price_list:
                # Count not predicting a price for a product that has no price as correct
                image_correct_total += 1

        total_prices_predicted += image_predicted
        total_predicted_prices_correct += image_correct_predicted
        total_prices_correct += image_correct_total
        image_pred_accuracy = (
            image_correct_predicted / image_predicted if image_predicted > 0 else 0
        )
        image_gt_accuracy = image_correct_total / img_path_to_gt_count[filename]
        img_err_gen = (
            image_errors_from_gen_model / image_predicted if image_predicted > 0 else 0
        )
        img_err_att = (
            image_errors_from_attribution / image_predicted
            if image_predicted > 0
            else 0
        )

        if image_pred_accuracy == 1 or image_gt_accuracy == 1:
            continue
        print(f"Estimated image errors from generative model: {img_err_gen}")
        print(f"Estimated image errors from attribution: {img_err_att}")
        print(f"Image Accuracy (pred only): {image_pred_accuracy}")
        print(f"Image Accuracy (total): {image_gt_accuracy}")
        print("All gt prices:", gt_prices[filename]["all"])
        print("All predictions:", set(valid_predictions))

        print("\nCorrectly Attributed:")
        for a in sorted(attributed, key=lambda x: x["label"]):
            print(a)
        print("\nMissed:")
        for m in sorted(missed, key=lambda x: x["label"]):
            print(m)
        print("\nNot Predicted:")
        for np in sorted(
            [label for label in labels_not_predicted if label[1]], key=lambda x: x[0]
        ):
            print(np)

    pred_accuracy = (
        total_predicted_prices_correct / total_prices_predicted
        if total_prices_predicted > 0
        else 0
    )
    total_accuracy = total_prices_correct / total_gt_prices
    tot_err_gen = (
        total_price_error_from_gen_model / total_prices_predicted
        if total_prices_predicted > 0
        else 0
    )
    tot_err_att = (
        total_price_error_from_attribution / total_prices_predicted
        if total_prices_predicted > 0
        else 0
    )
    print(f"Total Predicted: {total_prices_predicted}")
    print(f"Total Ground Truth: {total_gt_prices}")
    print(f"Estimated error % from generative model: {tot_err_gen}")
    print(f"Estimated error % from attribution: {tot_err_att}")
    print(f"Accuracy (predicted only): {pred_accuracy}")
    print(f"Accuracy (total): {total_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Price Attribution")
    parser.add_argument(
        "--dataset-dir", type=str, help="Path to the validation dataset directory"
    )
    args = parser.parse_args()
    validate(args.dataset_dir)
