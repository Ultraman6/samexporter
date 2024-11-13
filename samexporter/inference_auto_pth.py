import argparse
import os
import sys
import json
import pathlib
import time

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.amg import write_masks_to_folder

sys.path.append(".")

import cv2
import numpy as np

from samexporter.sam_onnx import SegmentAnythingONNX
from samexporter.sam2_onnx import SegmentAnything2ONNX


def str2bool(v):
    return v.lower() in ("true", "1")


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--image",
    type=str,
    default="E:\GitHub\samexporter\images/truck.jpg",
    help="Path to the ONNX encoder model",
)
argparser.add_argument(
    "--checkpoint",
    type=str,
    default="E:\Models\samexporter\sam\origin\sam_vit_b_01ec64.pth",
    help="Path to the ONNX encoder model",
)
argparser.add_argument(
    "--model-type",
    type=str,
    default='vit_b',
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. "
    "Which type of SAM model to export.",
)
argparser.add_argument(
    "--prompt",
    type=str,
    default="E:\GitHub\samexporter\images/truck_prompt.json",
    help="Path to the image",
)
argparser.add_argument(
    "--output",
    type=str,
    default='E:\GitHub\samexporter\output1',
    help="Path to the output image",
)
argparser.add_argument(
    "--output_mode",
    type=str,
    default="binary_mask",
    choices=["binary_mask", "coco_rle"],
)
argparser.add_argument(
    "--show",
    type=bool,
    default=True,
)
args = argparser.parse_args()


image = cv2.imread(args.image)
prompt = json.load(open(args.prompt))
points = []
labels = []
boxes = []
for mark in prompt:
    if mark["type"] == "point":
        points.append(mark["data"])
        labels.append(mark["label"])
    elif mark["type"] == "rectangle":
        boxes.append(mark["data"])

points, labels, boxes = np.array(points), np.array(labels), np.array(boxes)
sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
sam.to("cuda")
model = SamAutomaticMaskGenerator(sam, output_mode=args.output_mode)
# 记录开始时间
start_time = time.time()
# 执行模型预测
masks = model.generate(image)
# 记录结束时间
end_time = time.time()
# 计算并打印耗时
elapsed_time = end_time - start_time
print(f"Time taken for model.predict_masks: {elapsed_time:.4f} seconds")

base = os.path.basename(args.image)
base = os.path.splitext(base)[0]
save_base = str(os.path.join(args.output, base))
if args.output_mode == "binary_mask":
    os.makedirs(save_base, exist_ok=True)
    write_masks_to_folder(masks, save_base)
else:
    save_file = save_base + ".json"
    with open(save_file, "w") as f:
        json.dump(masks, f)