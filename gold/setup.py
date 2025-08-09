import argparse
import math

import numpy as np
import pandas as pd
import os
import shutil

from pandas import DataFrame
from tqdm import tqdm
from google.cloud import storage
from sklearn.model_selection import train_test_split

from .data_setup_utils import (
    get_yaml,
    GoldGeneratorConfig,
    write_label_files,
    write_names_file,
    add_to_cls_column,
)

# SOME DEFAULTS
SEED = 1998
# NUM UNIQUE IMAGES = 6869
# NUM UNIQUE CLASSES = 6311 (6310 PRODUCTS)
BOX_THRESH = 100
NUM_IMG_TO_LABEL = 100  # ???
TAG_UPC = "000000000017"


def get_file_paths(directory: str):
    """
    Recursively gets the paths of all files starting from a directory.

    Args:
        directory (str): The root directory to start searching.

    Returns:
        list: A list of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def init_dirs():
    for prefix in ["product", "price-tag"]:
        for _type in ["labels", "images"]:
            for set_type in ["train", "val", "test"]:
                os.makedirs(
                    f"./gold/detection/{prefix}/data/{_type}/{set_type}", exist_ok=True
                )


def split_df(
    df: pd.DataFrame, config: GoldGeneratorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a train, val, test split on a dataframe using the `bucket_path` column to split.
    """
    bucket_path = df.bucket_file_name.unique()
    train_vals, temp_vals = train_test_split(
        bucket_path, test_size=0.2, random_state=config.seed
    )
    val_vals, test_vals = train_test_split(
        temp_vals, test_size=0.5, random_state=config.seed
    )

    train_df = df[df.bucket_file_name.isin(train_vals)]
    val_df = df[df.bucket_file_name.isin(val_vals)]
    test_df = df[df.bucket_file_name.isin(test_vals)]

    return train_df, val_df, test_df


def download_images(df: pd.DataFrame, download_dir: str):
    """
    Downloads a set of images to the given directory
    """
    os.makedirs(download_dir, exist_ok=True)
    items = df.to_dict("records")
    client = storage.Client()
    data = ((item["id"], item["bucket_file_name"]) for item in items)
    for img_id, bucket_file_name in tqdm(
        data, total=len(items), desc="Downloading images..."
    ):
        destination_file_name = os.path.join(download_dir, f"{img_id}.jpg")
        if not os.path.exists(destination_file_name):
            bucket_name, blob_name = bucket_file_name.split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(destination_file_name)


def setup_detection(df: pd.DataFrame, download_dir: str, config: GoldGeneratorConfig):
    (product_dfs, tag_dfs) = (
        ("product", split_df(df[df.ml_label_name != TAG_UPC], config)),
        ("price-tag", split_df(df[df.ml_label_name == TAG_UPC], config)),
    )
    if config.write_label_files:
        for dataset_type, (train_df, val_df, test_df) in [product_dfs, tag_dfs]:
            prefix = os.path.join("./gold", "detection", dataset_type)
            classes = (
                [TAG_UPC]
                if dataset_type == "price-tag"
                else list(df.ml_label_name.unique())
            )
            common_args = {
                "classes": classes,
                "orig_img_path_prefix": download_dir,
                "prefix": prefix,
            }
            write_label_files(
                **common_args,
                dataset=train_df,
                set_type="train",
            )
            write_label_files(
                **common_args,
                dataset=val_df,
                set_type="val",
            )
            write_label_files(
                **common_args,
                dataset=test_df,
                set_type="test",
            )
            write_names_file(
                classes, filepath=f"./gold/detection/{dataset_type}/config.yaml"
            )
    return product_dfs, tag_dfs


def get_path_df(dtype: str) -> pd.DataFrame:
    df: DataFrame = pd.DataFrame(
        {
            "img_path": [
                path
                for path in get_file_paths(f"./gold/detection/{dtype}/data/images")
                if path.endswith(".jpg")
            ],
        }
    )
    df["label_path"] = df.img_path.str.replace("images", "labels").str.replace(
        ".jpg", ".txt"
    )
    df["id"] = df.img_path.str.split("/").str[-1].str.replace(".jpg", "")
    return df


def setup_association(
    product_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tag_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    config: GoldGeneratorConfig,
):
    os.makedirs("./gold/association", exist_ok=True)
    prefixes = ["train", "val", "test"]
    sizes = [
        # 80 / 10 / 10 split
        math.floor(config.num_img_to_label * 0.8),
        math.floor(config.num_img_to_label * 0.1),
        math.floor(config.num_img_to_label * 0.1),
    ]
    np.random.seed(config.seed)  # Set the seed for reproducibility
    # TODO: get paths of price tag annotations
    product_path_df = get_path_df("product")
    tag_path_df = get_path_df("price-tag")
    for i, (product_df, tag_df) in enumerate(zip(product_dfs, tag_dfs)):
        # Get samples that have both products and price tags
        ids_to_sample = list(set(product_df.id).intersection(set(tag_df.id)))
        ids_to_sample = np.random.choice(ids_to_sample, size=sizes[i], replace=False)
        df = pd.concat([product_df, tag_df])
        df_assoc = df[df.id.isin(ids_to_sample)]
        df_assoc.to_csv(f"./gold/association/{prefixes[i]}_dataset.csv", index=False)
        loop_path_df = pd.merge(
            product_path_df,
            tag_path_df,
            on="id",
            how="inner",
            suffixes=("_product", "_tag"),
        )
        loop_path_df = loop_path_df[loop_path_df.id.isin(ids_to_sample)]
        loop_path_df.to_csv(f"./gold/association/{prefixes[i]}_paths.csv", index=False)
        copy_association(loop_path_df)


def copy_association(path_df: pd.DataFrame):
    root = "./gold/association"
    for i, row in path_df.iterrows():
        _id = row.id
        plabel = row.label_path_product
        prlabel = row.label_path_tag
        pimg = row.img_path_product
        label_path = os.path.join(root, "labels", f"{_id}.txt")
        shutil.copy(pimg, os.path.join(root, "images", f"{_id}.jpg"))
        add_to_cls_column(plabel, label_path)

        # Read the contents of both files
        with open(label_path, "r") as pr_labels, open(prlabel, "r") as tag_labels:
            products = pr_labels.read()
            tags = tag_labels.read()

        # Combine the contents with a newline in between
        combined_contents = f"{products}\n{tags}"

        # Write the combined contents to the output file
        with open(label_path, "w") as output_file:
            output_file.write(combined_contents)


def main(config: GoldGeneratorConfig, download=False):
    df = pd.read_csv("./gold/csvs/products_and_tags.csv", dtype={"ml_label_name": str})
    df = df[~df.ml_label_name.isin(["Unidentified Product", "000000000018"])]
    df["id"] = df.bucket_file_name.str.split("/").str[-1].str.replace(".jpg", "")

    classes = df.ml_label_name.value_counts().reset_index()
    classes_above_box_thresh = classes[classes["count"] >= config.box_thresh]
    df = df[df.ml_label_name.isin(classes_above_box_thresh.ml_label_name)]
    df.to_csv("./gold/csvs/dataset.csv", index=False)

    download_dir = "./gold/images"
    if download:
        download_images(df[["bucket_file_name", "id"]].drop_duplicates(), download_dir)
    product_dfs, tag_dfs = setup_detection(df, download_dir, config)
    setup_association(product_dfs[1], tag_dfs[1], config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--download-images", action="store_true", default=False)
    args = parser.parse_args()
    config_dict = get_yaml(args.config)
    init_dirs()
    main(
        GoldGeneratorConfig(
            seed=config_dict["seed"],
            box_thresh=config_dict["box_thresh"],
            num_img_to_label=config_dict["association"]["num_img_to_label"],
            write_label_files=config_dict["write_label_files"],
        ),
        download=args.download_images,
    )

# TODO:
#  - Remove all boxes of products with less than BOX_THRESH boxes
#  - Train product detector on all images with a product (420 classes)
#  - Train price tag detector on all images with a price tag (1 class)
#    - If we trained with only model, the price tag class would DOMINATE (about 1/2 the examples)
#  - Mark on images that have BOTH product boxes and tag boxes
#    - Do random train/test split on NUM_IMG_TO_LABEL boxes
#  Output:
#    gold/
#      setup.py <= This script
#      images/ <= Original download location of images
#      csvs/
#        products_and_tags.csv  <= downloaded from BQ
#        dataset.csv  <= dataset used to construct "detection" and "association"
#      detection/
#        [type]/  <= "product" | "price-tag"
#          config.yaml
#          data/
#            labels/
#              train/
#              val/
#              test/
#            images/
#              train/
#              val/
#              test/
#      association/
#        dataset.csv  <= dataset of images we will label
#        paths.csv    <= contains file paths to both the image and annotation of the images inside of `dataset.csv`
#
