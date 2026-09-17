
"""Export the complete SG-NCA Cholec inference pipeline for Android.

Example:
    python export_model.py \
      --checkpoint /path/to/dazzling-carnation-75 \
      --output-dir app/app/src/main/assets/models \
      --validation-frames /path/to/frame1.png /path/to/frame2.png ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

from data.cholec_scene_graph import CholecSceneGraphDataset
from export.mobile_wrappers import MobileFrameWrapper, MobileRelationWrapper
from training_utils.trainer import BaseTrainer


IMAGE_SIZE = 256
MASK_THRESHOLD = 0.5
AREA_THRESHOLD = 150
RELATION_THRESHOLD = 0.5
SMOOTHING_WINDOW = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-frames",
        type=Path,
        nargs="*",
        default=[],
        help="Eight chronological 256x256 RGB frames used for parity validation.",
    )
    return parser.parse_args()


def load_frame(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_metadata(trainer) -> dict:
    dataset = CholecSceneGraphDataset
    class_ids = sorted(dataset.SEG_LABELS.inverse.keys())
    rng = np.random.default_rng(0)
    colors = rng.integers(50, 255, size=(len(class_ids), 3), dtype=np.uint8)
    return {
        "schema_version": 1,
        "dataset": "Cholec",
        "checkpoint": trainer.run_name,
        "input": {
            "name": "image",
            "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "color_order": "RGB",
            "value_range": [0.0, 1.0],
        },
        "frame_outputs": {
            "segmentation_logits": [1, len(class_ids), IMAGE_SIZE, IMAGE_SIZE],
            "object_features": [1, len(class_ids), 165],
            "presence": [1, len(class_ids)],
        },
        "relation_input": {
            "name": "object_context",
            "shape": [len(class_ids), trainer.relation_config["num_frames"], 165],
        },
        "relation_output": {
            "name": "relation_logits",
            "shape": [len(dataset.POSSIBLE_PAIRS), len(dataset.VERBS) - 1],
        },
        "classes": [
            {
                "id": class_id,
                "name": dataset.SEG_LABELS.inverse[class_id],
                "color": colors[index].tolist(),
            }
            for index, class_id in enumerate(class_ids)
        ],
        "verbs": [dataset.VERBS.inverse[index] for index in range(len(dataset.VERBS) - 1)],
        "possible_pairs": [list(pair) for pair in dataset.POSSIBLE_PAIRS],
        "fps": dataset.FPS,
        "num_frames": trainer.relation_config["num_frames"],
        "time_span_seconds": trainer.relation_config["time_span"],
        "temporal_offsets": [-25, -21, -18, -14, -11, -7, -4, 0],
        "mask_threshold": MASK_THRESHOLD,
        "area_threshold": AREA_THRESHOLD,
        "relation_threshold": RELATION_THRESHOLD,
        "smoothing_window": SMOOTHING_WINDOW,
    }


def assert_model_contract(path: Path, expected_inputs: dict, expected_outputs: dict) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)

    def value_shapes(values):
        result = {}
        for value in values:
            result[value.name] = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        return result

    actual_inputs = value_shapes(model.graph.input)
    actual_outputs = value_shapes(model.graph.output)
    if actual_inputs != expected_inputs:
        raise AssertionError(f"Unexpected inputs for {path}: {actual_inputs}")
    if actual_outputs != expected_outputs:
        raise AssertionError(f"Unexpected outputs for {path}: {actual_outputs}")


def export_models(args: argparse.Namespace) -> None:
    trainer = BaseTrainer.load_checkpoint(str(args.checkpoint))
    if trainer.dataset_class.__name__ != "CholecDataset":
        raise ValueError("The v1 mobile export supports only the Cholec checkpoint.")
    if not trainer.is_relation_trained:
        raise ValueError("The checkpoint does not contain a trained relation classifier.")

    trainer.network.eval()
    trainer.relation_classifier.eval()
    trainer.network.merge_ncas()

    dataset = CholecSceneGraphDataset
    class_ids = sorted(dataset.SEG_LABELS.inverse.keys())
    channel_order = sum(trainer.classes, [])
    frame_wrapper = MobileFrameWrapper(
        trainer.network,
        channel_order=channel_order,
        class_ids=class_ids,
        image_size=IMAGE_SIZE,
        mask_threshold=MASK_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
    ).eval()
    relation_wrapper = MobileRelationWrapper(
        trainer.relation_classifier,
        class_ids=class_ids,
        possible_pairs=dataset.POSSIBLE_PAIRS,
    ).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_model_path = args.output_dir / "frame_model.onnx"
    relation_model_path = args.output_dir / "relation_model.onnx"
    metadata_path = args.output_dir / "model_config.json"

    dummy_image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    dummy_context = torch.zeros(
        len(class_ids),
        trainer.relation_config["num_frames"],
        165,
        dtype=torch.float32,
    )
    torch.onnx.export(
        frame_wrapper,
        dummy_image,
        frame_model_path,
        opset_version=17,
        input_names=["image"],
        output_names=["segmentation_logits", "object_features", "presence"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    torch.onnx.export(
        relation_wrapper,
        dummy_context,
        relation_model_path,
        opset_version=17,
        input_names=["object_context"],
        output_names=["relation_logits"],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    metadata = build_metadata(trainer)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    assert_model_contract(
        frame_model_path,
        {"image": [1, 3, IMAGE_SIZE, IMAGE_SIZE]},
        {
            "segmentation_logits": [1, len(class_ids), IMAGE_SIZE, IMAGE_SIZE],
            "object_features": [1, len(class_ids), 165],
            "presence": [1, len(class_ids)],
        },
    )
    assert_model_contract(
        relation_model_path,
        {"object_context": [len(class_ids), trainer.relation_config["num_frames"], 165]},
        {"relation_logits": [len(dataset.POSSIBLE_PAIRS), len(dataset.VERBS) - 1]},
    )

    if args.validation_frames:
        if len(args.validation_frames) != trainer.relation_config["num_frames"]:
            raise ValueError("Exactly eight validation frames are required.")
        validate_parity(
            frame_wrapper,
            relation_wrapper,
            frame_model_path,
            relation_model_path,
            args.validation_frames,
        )

    print(f"Exported {frame_model_path}")
    print(f"Exported {relation_model_path}")
    print(f"Exported {metadata_path}")


def validate_parity(
    frame_wrapper: MobileFrameWrapper,
    relation_wrapper: MobileRelationWrapper,
    frame_model_path: Path,
    relation_model_path: Path,
    validation_frames: list[Path],
) -> None:
    frames = torch.stack([load_frame(path) for path in validation_frames])
    with torch.inference_mode():
        torch_logits, torch_features, torch_presence = frame_wrapper(frames)
        torch_context = torch_features.permute(1, 0, 2)
        torch_relations = relation_wrapper(torch_context)

    frame_session = ort.InferenceSession(str(frame_model_path), providers=["CPUExecutionProvider"])
    relation_session = ort.InferenceSession(str(relation_model_path), providers=["CPUExecutionProvider"])
    ort_logits = []
    ort_features = []
    ort_presence = []
    for frame in frames.numpy():
        outputs = frame_session.run(None, {"image": frame[None]})
        ort_logits.append(outputs[0][0])
        ort_features.append(outputs[1][0])
        ort_presence.append(outputs[2][0])

    ort_context = np.stack(ort_features, axis=1)
    ort_relations = relation_session.run(None, {"object_context": ort_context})[0]

    np.testing.assert_allclose(np.stack(ort_logits), torch_logits.numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(np.stack(ort_features), torch_features.numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_array_equal(np.stack(ort_presence), torch_presence.numpy())
    np.testing.assert_allclose(ort_relations, torch_relations.numpy(), atol=1e-3, rtol=1e-3)
    print("PyTorch/ONNX parity passed (atol=1e-3, rtol=1e-3).")


if __name__ == "__main__":
    export_models(parse_args())
