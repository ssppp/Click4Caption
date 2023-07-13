import os
import json
import random

from PIL import Image
import numpy as np

from click4caption.datasets.datasets.base_dataset import BaseDataset
from click4caption.datasets.datasets.caption_datasets import CaptionDataset


class VGDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, num_regions, image_size):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.location = location
        self.num_regions = num_regions  # select num_regions regions each image for training
        self.image_size = image_size
        print(f"==> dataset image_size={image_size}")

        self.iid2ann = {}  # image id to annotation
        with open(os.path.join(self.location, "region_descriptions.json"), "r") as fp:
            r_ann = json.load(fp)
        for ann in r_ann:
            self.iid2ann[ann["id"]] = {"regions": ann["regions"]}

        with open(os.path.join(self.location, "image_data.json"), "r") as fp:
            i_ann = json.load(fp)
        for ann in i_ann:
            url_split = ann["url"].split("/")
            self.iid2ann[ann["image_id"]]["image_path"] = os.path.join(self.location, url_split[-2], url_split[-1])
        
        self.iid_list = list(self.iid2ann.keys())

    def __len__(self):
        return len(self.iid_list)

    def _process_coordinate(self, x, y, w, h, img_w, img_h, resize_size):
        w_scale = resize_size / img_w
        h_scale = resize_size / img_h
        left_x = x * w_scale
        left_y = y * h_scale
        right_x = (x + w) * w_scale
        right_y = (y + h) * h_scale
        return np.array([[left_x, left_y], [right_x, right_y]])
    
    def _is_valid(self, r, img_w, img_h):
        if len(r["phrase"].strip()) == 0:
            return False
        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
        bbox = [x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h]
        if min(bbox) < 0 or max(bbox) > 1:  # special for vg
            return False
        return True

    def __getitem__(self, index):
        iid = self.iid_list[index]
        image_path = self.iid2ann[iid]["image_path"]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        image = self.vis_processor(image)

        num_regions = self.num_regions
        valid_regions = [r for r in self.iid2ann[iid]["regions"] if self._is_valid(r, img_w, img_h)]
        regions = random.choices(valid_regions, k=num_regions)
        bboxes = np.empty((num_regions, 2, 2))
        r_captions = []
        for i, r in enumerate(regions):
            cap = r["phrase"].strip()
            if cap[0].islower():
                cap = cap[0].upper() + cap[1:]
            if cap[-1] != ".":
                cap += "."
            r_captions.append(cap)

            bboxes[i] = self._process_coordinate(r["x"], r["y"], r["width"], r["height"], img_w, img_h, resize_size=self.image_size)

        return {
            "image": image,
            "text_input": r_captions,
            "bbox": bboxes,
        }
