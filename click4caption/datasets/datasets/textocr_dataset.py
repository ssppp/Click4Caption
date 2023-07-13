import os
import json
import random

from PIL import Image
import numpy as np

from click4caption.datasets.datasets.base_dataset import BaseDataset
from click4caption.datasets.datasets.caption_datasets import CaptionDataset


class TextOCRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, num_regions, image_size):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.location = location
        self.num_regions = num_regions  # select num_regions regions each image for training
        self.image_size = image_size
        print(f"==> dataset image_size={image_size}")

        with open(os.path.join(self.location, "TextOCR_0.1_train.json"), "r") as fp:
            self.ann = json.load(fp)
        self.img_id_list = list(self.ann["imgs"].keys())

    def __len__(self):
        return len(self.img_id_list)

    def _process_coordinate(self, x, y, w, h, img_w, img_h, resize_size):
        w_scale = resize_size / img_w
        h_scale = resize_size / img_h
        left_x = x * w_scale
        left_y = y * h_scale
        right_x = (x + w) * w_scale
        right_y = (y + h) * h_scale
        return np.array([[left_x, left_y], [right_x, right_y]])

    def _is_valid(self, region_id, img_w, img_h):
        if self.ann["anns"][region_id]["utf8_string"] == '.':
            return False
        x, y, w, h = self.ann["anns"][region_id]["bbox"]
        if w * 224 / img_w < 5 or h * 224 / img_h < 5:  # skip the small text
            return False
        return True
    
    def __getitem__(self, index):
        while True:
            img_id = self.img_id_list[index]
            image_path = os.path.join(self.location, "train_images", os.path.basename(self.ann["imgs"][img_id]["file_name"]))
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
            image = self.vis_processor(image)

            num_regions = self.num_regions
            valid_regions = [r_id for r_id in self.ann["imgToAnns"][img_id] if self._is_valid(r_id, img_w, img_h)]
            if len(valid_regions) < num_regions // 2:
                index = random.randint(0, len(self)-1)
                continue
            
            regions = random.choices(valid_regions, k=num_regions)
            bboxes = np.empty((num_regions, 2, 2))
            r_captions = []
            for i, r_id in enumerate(regions):
                r = self.ann["anns"][r_id]
                cap = r["utf8_string"]
                x, y, w, h = r['bbox']
                r_captions.append(cap)
                bboxes[i] = self._process_coordinate(x, y, w, h, img_w, img_h, resize_size=self.image_size)
            
            break
        
        bboxes = bboxes.clip(min=0, max=self.image_size)  # special for textocr

        return {
            "image": image,
            "text_input": r_captions,
            "bbox": bboxes,
        }
