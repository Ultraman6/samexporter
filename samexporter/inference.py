import argparse
import sys
import json
import pathlib

sys.path.append(".")

import cv2
import numpy as np

from samexporter.sam_onnx import SegmentAnythingONNX
from samexporter.sam2_onnx import SegmentAnything2ONNX


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
    default='E:\GitHub\samexporter/results/trunk.jpg',
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
    model = SegmentAnythingONNX(
        args.encoder_model,
        args.decoder_model,
    )
elif args.sam_variant == "sam2":
    model = SegmentAnything2ONNX(
        args.encoder_model,
        args.decoder_model,
    )

image = cv2.imread(args.image)
prompt = json.load(open(args.prompt))

embedding = model.encode(image)

masks = model.predict_masks(embedding, prompt)

# Merge masks
mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
for m in masks[0, :, :, :]:
    mask[m > 0.5] = [255, 0, 0]

# Binding image and mask
visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

# Draw the prompt points and rectangles.
for p in prompt:
    if p["type"] == "point":
        color = (
            (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
        )  # green for positive, red for negative
        cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
    elif p["type"] == "rectangle":
        cv2.rectangle(
            visualized,
            (p["data"][0], p["data"][1]),
            (p["data"][2], p["data"][3]),
            (0, 255, 0),
            2,
        )

if args.output is not None:
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, visualized)

if args.show:
    cv2.imshow("Result", visualized)
    cv2.waitKey(0)
