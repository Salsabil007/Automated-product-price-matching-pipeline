import os.path
import shutil
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yaml


def get_box_properties(min_x, max_x, min_y, max_y):
    # Calculate the center
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # Calculate width and height
    width = max_x - min_x
    height = max_y - min_y

    return cx, cy, width, height


def write_label_files(
    classes: list[str],
    orig_img_path_prefix: str,
    dataset: pd.DataFrame,
    set_type: Literal["train", "val", "test"],
    prefix: str = "./",
):
    for image_id, image in dataset.groupby(["id"]):
        if isinstance(image_id, tuple):
            image_id = image_id[0]
        orig_img_path = os.path.join(orig_img_path_prefix, f"{image_id}.jpg")
        target_img_path = os.path.join(
            prefix, "data", "images", set_type, f"{image_id}.jpg"
        )
        if os.path.exists(orig_img_path):
            shutil.copy(orig_img_path, target_img_path)

        label_path = os.path.join(prefix, "data", "labels", set_type, f"{image_id}.txt")
        with open(label_path, "w") as f:
            for i, row in image.iterrows():
                cx, cy, width, height = get_box_properties(
                    row.min_x, row.max_x, row.min_y, row.max_y
                )
                upc = row.ml_label_name
                index = classes.index(upc)
                annotation_str = f"{' '.join([str(index), str(cx), str(cy), str(width), str(height)])}\n"
                f.write(annotation_str)


def write_names_file(classes: list[str], filepath: str):
    with open(filepath, "w") as f:
        classes_lines = [f"  {idx}: {cls}" for idx, cls in enumerate(classes)]
        f.write(
            "\n".join(
                [
                    "path: ./data",
                    "train: images/train",
                    "val: images/val",
                    "#test: images/train",  # TODO: why is this commented out again...?
                    f"nc: {len(classes)}",
                    "names:",
                    *classes_lines,
                ]
            )
        )


def get_yaml(fpath) -> dict:
    """Load the given .yaml file as a Python dictionary."""
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def add_to_cls_column(label_path: str, output_path: str):
    with open(label_path, "r") as file:
        lines = file.readlines()

    # Process each line
    updated_lines = []
    for line in lines:
        parts = line.split()  # Split the line into parts
        if parts:  # Ensure the line is not empty
            parts[0] = str(int(parts[0]) + 1)  # Increment the first value
            updated_lines.append(" ".join(parts))  # Join the updated parts back

    # Write the updated lines to a new file
    with open(output_path, "w") as file:
        file.write("\n".join(updated_lines))


@dataclass
class GoldGeneratorConfig:
    seed: int
    box_thresh: int
    num_img_to_label: 100
    write_label_files: bool = True


"""
cat data/labels/train/0C058F76-A39B-4299-8D11-2601C2F4FE76.txt

818094005777 — Rockstar Original 16oz Can

9 0.5628903657197954 0.8011222998301191 0.08670745491981491 0.259240627288818
9 0.5833724737167361 0.167427333196004 0.11000056266784608 0.334854666392008
9 0.3642306208610535 0.16790239016215 0.11269414424896301 0.3358047803243
9 0.478669023513794 0.8051205237706505 0.09087104797363199 0.26472560564676895
9 0.24829106330871548 0.167951989173889 0.129950928688049 0.335903978347778
9 0.390046989917755 0.8058645486831665 0.09302399158477803 0.26494145393371493
9 0.6494240432977675 0.7974554459253945 0.09508727192878696 0.25416280428568494
9 0.3001192152500155 0.807081524531047 0.09623500108718896 0.26688178380330396
9 0.4755352199077605 0.1677334705988565 0.11911262273788492 0.335466941197713
9 0.6899910867214205 0.1656110445658365 0.11500128507614105 0.331222089131673
"""
