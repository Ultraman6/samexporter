import argparse
import sys
import json
import pathlib

from segment_anything import sam_model_registry, SamPredictor

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
    default='E:\GitHub\samexporter/results/trunk1.jpg',
    help="Path to the output image",
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
model = SamPredictor(sam)
model.set_image(image)
masks_np, iou_predictions_np, low_res_masks_np = model.predict(
    point_coords=points,
    point_labels=labels,
    box=boxes,
)

# Step 1: Create a blank mask (image size is [height, width, 3] for RGB mask)
mask = np.zeros((masks_np.shape[1], masks_np.shape[2], 3), dtype=np.uint8)

# Step 2: Iterate through the masks and apply a color mask for each one
for i in range(masks_np.shape[0]):
    mask_single = masks_np[i, :, :]
    mask[mask_single > 0.5] = [255, 0, 0]  # Red color for mask pixels

# Step 3: Bind the original image and the mask (overlay)
visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

# Step 4: Draw the prompt points and rectangles
for p in prompt:
    if p["type"] == "point":
        # Green for positive, red for negative
        color = (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
        cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
    elif p["type"] == "rectangle":
        cv2.rectangle(
            visualized,
            (p["data"][0], p["data"][1]),
            (p["data"][2], p["data"][3]),
            (0, 255, 0),
            2,
        )

# Step 5: Optionally save the visualized image
if args.output is not None:
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, visualized)

# Step 6: Optionally show the visualized image
if args.show:
    cv2.imshow("Result", visualized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

