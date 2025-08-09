import glob
import json
import os

import yaml
import argparse

# yolo command
# yolo predict model=/path/to/bev/4.13.1/product_detector.pt source=images/ save=True project=product_runs save_txt=True
# yolo predict model=/path/to/meta/0.0.10/meta.pt source=images/ save=True project=meta_runs save_txt=True classes=1


def preprocess(dataset_dir: str):
    with open(os.path.join(dataset_dir, "upc_to_name.json"), "r") as f:
        upcs_and_names = json.load(f)
        upc_to_name = {obj["upc_a"]: obj["name"] for obj in upcs_and_names}

    with open(os.path.join(dataset_dir, "model/product_names.yaml")) as f:
        product_names = yaml.safe_load(f)["names"]

    with open(os.path.join(dataset_dir, "model", "meta_names.yaml")) as f:
        meta_names = yaml.safe_load(f)["names"]

    product_label_paths = glob.glob(
        os.path.join(dataset_dir, "product_runs/predict/labels/*.txt")
    )
    img_dir = os.path.join(dataset_dir, "images")

    dataset = []
    all_bounding_boxes = []
    for product_label_path in product_label_paths:
        filename = product_label_path.split("/")[-1]
        img_path = os.path.join(img_dir, filename.replace("txt", "jpg")).replace(
            dataset_dir, ""
        )
        if img_path.startswith("/"):
            img_path = img_path[1:]
        meta_label_path = product_label_path.replace("product_runs", "meta_runs")
        dataset_item = {"img_path": img_path, "products": []}
        bounding_boxes = {
            "img_path": img_path,
            "product_bounding_boxes": [],
            "price_tag_bounding_boxes": [],
        }
        image_products = set()
        with open(product_label_path) as f:
            for line in f:
                values = line.strip().split()
                index = int(values[0])
                upc = product_names[index]
                if upc not in image_products:
                    # Save one instance of each product per image
                    product_name = upc_to_name.get(upc, "Unknown")
                    dataset_item["products"].append(
                        {
                            "upc": upc,
                            "name": product_name,
                            "price_info": {
                                "price": None,
                                "sale_price": None,
                                "multi_price": "",
                                "unit_price": None,
                            },
                        }
                    )
                    image_products.add(upc)

                mid_x, mid_y, w, h = [float(val) for val in values[1:]]
                min_x = mid_x - w / 2
                min_y = mid_y - h / 2
                max_x = mid_x + w / 2
                max_y = mid_y + h / 2
                bounding_boxes["product_bounding_boxes"].append(
                    {
                        "label": upc,
                        "min_x": min_x,
                        "min_y": min_y,
                        "max_x": max_x,
                        "max_y": max_y,
                    }
                )
            dataset.append(dataset_item)

        with open(meta_label_path) as f:
            for line in f:
                values = line.strip().split()
                index = int(values[0])
                label = meta_names[index]
                if label != "000000000017":
                    # Only add price tags
                    continue
                mid_x, mid_y, w, h = [float(val) for val in values[1:]]
                min_x = mid_x - w / 2
                min_y = mid_y - h / 2
                max_x = mid_x + w / 2
                max_y = mid_y + h / 2
                bounding_boxes["price_tag_bounding_boxes"].append(
                    {
                        "label": label,
                        "min_x": min_x,
                        "min_y": min_y,
                        "max_x": max_x,
                        "max_y": max_y,
                    }
                )
        all_bounding_boxes.append(bounding_boxes)

    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    with open(os.path.join(dataset_dir, "bounding_boxes.json"), "w") as f:
        json.dump(all_bounding_boxes, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()
    preprocess(args.dataset_dir)
