import argparse
import os
import sys
import json
import pathlib
import time

sys.path.append(".")

import cv2
import numpy as np

from samexporter.sam_onnx import SegmentAnythingONNX, SamAutomaticMaskGenerator
from samexporter.sam2_onnx import SegmentAnything2ONNX
from segment_anything.utils.amg import write_masks_to_folder

def str2bool(v):
    return v.lower() in ("true", "1")


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--encoder_model",
    type=str,
    default="E:\Models\samexporter\sam\export\sam_vit_b_01ec64.encoder.onnx",
    help="Path to the ONNX encoder model",
)
argparser.add_argument(
    "--decoder_model",
    type=str,
    default="E:\Models\samexporter\sam\export\sam_vit_b_01ec64.decoder.onnx",
    help="Path to the ONNX decoder model",
)
argparser.add_argument(
    "--output_mode",
    type=str,
    default="binary_mask",
    choices=["binary_mask", "coco_rle"],
)
argparser.add_argument(
    "--image",
    type=str,
    default="E:\GitHub\samexporter\images/truck.jpg",
    help="Path to the image",
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
    default='E:\GitHub\samexporter\output2',
    help="Path to the output image",
)
argparser.add_argument(
    "--show",
    type=bool,
    default=True,
)
argparser.add_argument(
    "--sam_variant",
    type=str,
    default="sam",
    help="Variant of SAM model. Options: sam, sam2",
)
args = argparser.parse_args()

model = None
if args.sam_variant == "sam":
    model = SamAutomaticMaskGenerator(
        args.encoder_model,
        args.decoder_model,
        output_mode=args.output_mode
    )
elif args.sam_variant == "sam2":
    model = SegmentAnything2ONNX(
        args.encoder_model,
        args.decoder_model,
    )

image = cv2.imread(args.image)


# 记录开始时间
start_time = time.time()
# 执行模型预测
masks = model.predict_masks(image)
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