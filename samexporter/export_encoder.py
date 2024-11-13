import torch
import onnx

from segment_anything import sam_model_registry
from samexporter.mobile_encoder.setup_mobile_sam import setup_model
from samexporter.onnx_utils import ImageEncoderOnnxModel
from onnx.external_data_helper import convert_model_to_external_data

import os
from tempfile import mkdtemp
import pathlib
import shutil
import argparse
import warnings

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="E:\Models\samexporter\sam\origin\sam_vit_b_01ec64.pth",
)

parser.add_argument(
    "--output",
    type=str,
    default="E:\Models\samexporter\sam\export\sam_vit_b_01ec64.encoder.onnx",
)

parser.add_argument(
    "--model-type",
    type=str,
    default='vit_b',
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. "
    "Which type of SAM model to export.",
)

parser.add_argument(
    "--use-preprocess",
    type=bool,
    default=True,
    help="Embed pre-processing into the model",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default='E:\Models\samexporter\sam\sam_vit_l_0b3195.encoder.quant.onnx'
)


parser.add_argument(
    "--gelu-approximate",
    type=bool,
    default=False,
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
    gelu_approximate: bool = False,
):
    print("Loading model...")
    if model_type == "mobile":
        checkpoint = torch.load(checkpoint, map_location="cpu")
        sam = setup_model()
        sam.load_state_dict(checkpoint, strict=True)
    else:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = ImageEncoderOnnxModel(
        model=sam,
        use_preprocess=use_preprocess,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = sam.image_encoder.img_size
    if use_preprocess:
        dummy_input = {
            "input_image": torch.randn(
                (image_size, image_size, 3), dtype=torch.float
            )
        }
        dynamic_axes = {
            "input_image": {0: "image_height", 1: "image_width"},
        }
    else:
        dummy_input = {
            "input_image": torch.randn(
                (1, 3, image_size, image_size), dtype=torch.float
            )
        }
        dynamic_axes = None

    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    onnx_base = os.path.splitext(os.path.basename(output))[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        if model_type == "vit_h":
            tmp_dir = mkdtemp()
            tmp_model_path = os.path.join(tmp_dir, f"{onnx_base}.onnx")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                tmp_model_path,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Combine the weights into a single file
            pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
            onnx_model = onnx.load(tmp_model_path)
            convert_model_to_external_data(
                onnx_model,
                all_tensors_to_one_file=True,
                location=f"{onnx_base}_data.bin",
                size_threshold=1024,
                convert_attribute=False,
            )

            # Save the model
            onnx.save(onnx_model, output)

            # Cleanup the temporary directory
            shutil.rmtree(tmp_dir)
        else:
            with open(output, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_input.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=list(dummy_input.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    if args.quantize_out is not None:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
