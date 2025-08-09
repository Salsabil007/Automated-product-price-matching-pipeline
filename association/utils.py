import json
import os
import random

import cv2


random.seed(1998)


def shuffle_lines(filepath: str):
    """
    Shuffles the lines of a label file (.txt), as to introduce randomness.
    :param filepath:
    :return:
    """
    with open(filepath, "r") as f:
        lines = list(map(lambda x: f"{x}\n" if "\n" not in x else x, f.readlines()))
        random.shuffle(lines)
    with open(filepath, "w") as f:
        f.writelines(lines)
    return len(lines)


def write_boxes_and_box_number_to_img(data_path: str):
    """
    Writes a new image with the boxes and box number in the middle
    for simplified labeling.
    Creates a new image directory images_processed/ and predefined .json files (json/ directory)
    for labeling.
    :param data_path: path of the dataset folder, with images/ and labels/
    :return:
    """
    os.makedirs(os.path.join(data_path, "images_processed"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "json"), exist_ok=True)
    images = sorted(
        list(filter(lambda x: x.endswith(".jpg"), os.listdir(f"{data_path}/images")))
    )
    labels = sorted(
        list(filter(lambda x: x.endswith(".txt"), os.listdir(f"{data_path}/labels")))
    )
    font = cv2.FONT_ITALIC
    font_scale = 1
    thickness = 4
    color = (0, 255, 0)

    with open(f"./label_set.json", "r") as f:
        label_set = json.load(f)

    for image_file, label_file in zip(images, labels):
        if image_file not in label_set:
            continue
        img_id = image_file.split(".")[0]
        num_boxes = shuffle_lines(os.path.join(data_path, "labels", label_file))

        with open(os.path.join(data_path, "json", f"{img_id}.json"), "w") as file:
            json.dump(
                {"len": num_boxes, "complete": False, "associations": {}},
                file,
                indent=2,
            )

        with open(os.path.join(data_path, "labels", label_file), "r") as labels:
            lines = labels.readlines()
            img = cv2.imread(f"{data_path}/images/{image_file}")
            img_height, img_width = img.shape[:2]
            for idx, line in enumerate(lines):
                cls, ctr_x, ctr_y, width, height = line.split()
                ctr_x, ctr_y = img_width * float(ctr_x), img_height * float(ctr_y)
                cv2.putText(
                    img,
                    str(idx),
                    (round(ctr_x) - 12, round(ctr_y)),
                    font,
                    font_scale,
                    color=color,
                    thickness=thickness,
                )
                draw_box(
                    img,
                    ctr_x,
                    ctr_y,
                    box_width=img_width * float(width),
                    box_height=img_height * float(height),
                    color=color,
                )
            cv2.imwrite(
                os.path.join(data_path, "images_processed", f"{img_id}.jpg"), img
            )


def get_box_coords(
    ctr_x: float,
    ctr_y: float,
    box_width: float,
    box_height: float,
    img_width: int = 1,
    img_height: int = 1,
):
    """
    Returns (x1, y1, x2, y2) box coordinates.
    """

    def _round(val):
        if img_width > 1 or img_height > 1:
            return val
        return round(val)

    x1, y1 = (_round(ctr_x - box_width // 2), _round(ctr_y - box_height // 2))
    x2, y2 = (_round(ctr_x + box_width // 2), _round(ctr_y + box_height // 2))
    return x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height


def draw_box(image, ctr_x, ctr_y, box_width, box_height, color):
    x1, y1, x2, y2 = get_box_coords(ctr_x, ctr_y, box_width, box_height)
    thickness = 4
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
