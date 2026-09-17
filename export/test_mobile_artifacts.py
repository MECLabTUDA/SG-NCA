from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "app" / "src" / "main" / "assets"
MODELS = ASSETS / "models"
DEMO = ASSETS / "demo"


class MobileArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((MODELS / "model_config.json").read_text())
        cls.demo = json.loads((DEMO / "demo_config.json").read_text())

    def test_metadata_and_onnx_contracts(self):
        self.assertEqual([-25, -21, -18, -14, -11, -7, -4, 0], self.metadata["temporal_offsets"])
        self.assertEqual(10, len(self.metadata["classes"]))
        self.assertEqual(14, len(self.metadata["possible_pairs"]))
        self.assertEqual(20, len(self.demo["target_frames"]))
        self.assertEqual(46, len(self.demo["available_frames"]))
        available = set(self.demo["available_frames"])
        for target in self.demo["target_frames"]:
            self.assertTrue(all(
                target + offset in available
                for offset in self.demo["temporal_offsets"]
            ))

        frame_model = onnx.load(MODELS / "frame_model.onnx")
        relation_model = onnx.load(MODELS / "relation_model.onnx")
        onnx.checker.check_model(frame_model)
        onnx.checker.check_model(relation_model)
        self.assertEqual(["image"], [value.name for value in frame_model.graph.input])
        self.assertEqual(
            ["segmentation_logits", "object_features", "presence"],
            [value.name for value in frame_model.graph.output],
        )
        self.assertEqual(["object_context"], [value.name for value in relation_model.graph.input])
        self.assertEqual(["relation_logits"], [value.name for value in relation_model.graph.output])

    def test_bundled_sequence_has_expected_top_relation(self):
        frame_session = ort.InferenceSession(
            str(MODELS / "frame_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        relation_session = ort.InferenceSession(
            str(MODELS / "relation_model.onnx"),
            providers=["CPUExecutionProvider"],
        )

        feature_frames = []
        current_presence = None
        target = self.demo["target_frames"][0]
        frame_numbers = [target + offset for offset in self.demo["temporal_offsets"]]
        for frame_number in frame_numbers:
            filename = f"{frame_number:08d}.png"
            image = np.asarray(Image.open(DEMO / "frames" / filename).convert("RGB"), dtype=np.float32)
            nchw = np.transpose(image / 255.0, (2, 0, 1))[None]
            _, features, presence = frame_session.run(None, {"image": nchw})
            feature_frames.append(features[0])
            current_presence = presence[0]

        context = np.stack(feature_frames, axis=1)
        logits = relation_session.run(None, {"object_context": context})[0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        classes = {item["id"]: item["name"] for item in self.metadata["classes"]}
        candidates = []
        for pair_index, (subject, object_) in enumerate(self.metadata["possible_pairs"]):
            if not current_presence[subject - 1] or not current_presence[object_ - 1]:
                continue
            for verb_index, score in enumerate(probabilities[pair_index]):
                if score > self.metadata["relation_threshold"]:
                    candidates.append((
                        float(score),
                        classes[subject],
                        self.metadata["verbs"][verb_index],
                        classes[object_],
                    ))

        strongest = max(candidates)
        self.assertEqual(("grasper", "retract", "gallbladder"), strongest[1:])
        self.assertAlmostEqual(0.938, strongest[0], delta=0.002)


if __name__ == "__main__":
    unittest.main()
