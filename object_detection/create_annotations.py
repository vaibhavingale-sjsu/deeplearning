import os
from pathlib import Path

import fire
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
from tqdm import tqdm

CATEGORY_ID_TO_NAME = {
    "0": "ignore",
    "1": "pedestrian",
    "2": "people",
    "3": "bicycle",
    "4": "car",
    "5": "van",
    "6": "truck",
    "7": "tricycle",
    "8": "awning-tricycle",
    "9": "bus",
    "10": "motor",
    "11": "others",
}

CATEGORY_ID_REMAPPING = {
    "1": "0",
    "2": "1",
    "3": "2",
    "4": "3",
    "5": "4",
    "6": "5",
    "7": "6",
    "8": "7",
    "9": "8",
    "10": "9",
}

NAME_TO_COCO_CATEGORY = {
    "pedestrian": {"name": "pedestrian", "supercategory": "person"},
    "people": {"name": "people", "supercategory": "person"},
    "bicycle": {"name": "bicycle", "supercategory": "bicycle"},
    "car": {"name": "car", "supercategory": "car"},
    "van": {"name": "van", "supercategory": "truck"},
    "truck": {"name": "truck", "supercategory": "truck"},
    "tricycle": {"name": "tricycle", "supercategory": "motor"},
    "awning-tricycle": {"name": "awning-tricycle", "supercategory": "motor"},
    "bus": {"name": "bus", "supercategory": "bus"},
    "motor": {"name": "motor", "supercategory": "motor"},
}


import numpy as np

# import pandas as pd
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json

# Define your class dictionary depending on how many classes you have.
# Here, I'm assuming '0' is for potholes. Adjust as per your dataset.
# class_dic = {0: "pothole"}
class_dic = CATEGORY_ID_TO_NAME


def yolo_to_coco(x_center, y_center, w, h, image_w, image_h):
    # Convert YOLO format to COCO format
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w) / 2
    y1 = ((2 * y_center * image_h) - h) / 2
    return [x1, y1, w, h]


def convert_to_COCO(image_path, json_name):
    coco = Coco(image_id_setting="manual")
    for key, item in class_dic.items():
        coco.add_category(CocoCategory(id=key, name=item))

    # Adjust the glob pattern as per your folder structure
    image_path_list = list(Path(image_path).glob("*.jpg"))
    label_path_list = [
        Path(str(p).replace("images", "labels").replace(".jpg", ".txt"))
        for p in image_path_list
    ]

    print(f"The number of images: {len(image_path_list)}")
    print(f"The number of labels: {len(label_path_list)}")

    no_annotation = 0
    for img_path, l_path in tqdm(
        zip(image_path_list, label_path_list), total=len(image_path_list)
    ):
        ImageHeight, Imagewidth = Image.open(img_path).size

        # base_name = os.path.basename(annotation_filepath)
        name, ext = os.path.splitext(img_path.name)
        image_id = name
        # print(image_id)

        if os.path.isfile(l_path):
            coco_image = CocoImage(
                file_name=str(img_path.name),
                height=ImageHeight,
                width=Imagewidth,
                id=str(image_id),
            )
            with open(str(l_path)) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    obj_class, x_center, y_center, width_yolo, height_yolo = line.split(
                        " "
                    )
                    bbox = yolo_to_coco(
                        float(x_center),
                        float(y_center),
                        float(width_yolo),
                        float(height_yolo),
                        Imagewidth,
                        ImageHeight,
                    )
                    # cat_name = class_dic[int(obj_class)]
                    cat_name = int(obj_class)
                    coco_image.add_annotation(
                        CocoAnnotation(
                            bbox=bbox,
                            category_id=int(obj_class),
                            category_name=cat_name,
                        )
                    )
            coco.add_image(coco_image)
        else:
            no_annotation += 1

    print(f"The number of images that don't have a label: {no_annotation}")
    save_path = json_name
    save_json(data=coco.json, save_path=save_path)
    print("COCO Json File Created.")


# Specify your dataset paths
image_path = "/Users/vaibhav/Desktop/visdrone_dataset/VisDrone2019-DET-val/images"
json_name_file = "/Users/vaibhav/Desktop/visdrone_dataset/VisDrone2019-DET-val/gt1.json"
convert_to_COCO(image_path, json_name_file)
