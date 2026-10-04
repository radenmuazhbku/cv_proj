from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cv_proj.sam3_dfff import (
    average_precision_at_iou,
    bbox_iou,
    box_proxy_mask_metrics,
    prepare_manifest,
)
from scripts.infer_sam3_dfff import _expand_xyxy_box, _parse_box_expansion, _predict_one


class Sam3DfffPreparationTests(unittest.TestCase):
    def test_prompt_box_expansion_parsing(self) -> None:
        self.assertEqual(_parse_box_expansion("1.5"), ("ratio", 1.5))
        self.assertEqual(_parse_box_expansion("2."), ("ratio", 2.0))
        self.assertEqual(_parse_box_expansion("12"), ("pixels", 12.0))
        self.assertEqual(_parse_box_expansion("12px"), ("pixels", 12.0))

    def test_prompt_box_expansion_geometry_and_clipping(self) -> None:
        self.assertEqual(
            _expand_xyxy_box([10, 20, 30, 60], (100, 100), ("ratio", 1.5)),
            [5.0, 10.0, 35.0, 70.0],
        )
        self.assertEqual(
            _expand_xyxy_box([2, 3, 20, 24], (25, 25), ("pixels", 10.0)),
            [0.0, 0.0, 25.0, 25.0],
        )

    def _write_split(
        self,
        root: Path,
        split: str,
        image_ids: list[int],
        annotations: list[dict[str, Any]],
    ) -> None:
        image_root = root / split / "images"
        annotation_root = root / split / "annotations"
        image_root.mkdir(parents=True)
        annotation_root.mkdir()
        images = []
        for image_id in image_ids:
            name = f"{image_id}.png"
            Image.new("RGB", (32, 24), color="white").save(image_root / name)
            images.append({"id": image_id, "file_name": f"images/{name}", "width": 32, "height": 24})
        categories = [
            {"id": 1, "name": "NT", "supercategory": "object"},
            {"id": 2, "name": "nasal bone", "supercategory": "object"},
        ]
        (annotation_root / "instances.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories}),
            encoding="utf-8",
        )

    def test_manifest_keeps_query_boxes_separate_and_images_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            self._write_split(
                root,
                "set1",
                [1, 2, 3],
                [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 8, 7]},
                    {"id": 2, "image_id": 2, "category_id": 2, "bbox": [4, 5, 6, 8]},
                ],
            )
            self._write_split(
                root,
                "internal",
                [11, 12],
                [
                    {"id": 11, "image_id": 11, "category_id": 1, "bbox": [1, 2, 9, 8]},
                    {"id": 12, "image_id": 12, "category_id": 2, "bbox": [2, 3, 7, 6]},
                ],
            )
            output = Path(directory) / "episodes"
            manifest, evaluation = prepare_manifest(
                root,
                output,
                support_split="set1",
                query_split="internal",
                class_names=("NT", "nasal bone"),
                seed=5,
                copy_images=True,
            )

            for class_name, records in manifest["classes"].items():
                self.assertEqual(len(records["supports"]), 1)
                self.assertEqual(len(records["queries"]), 2)
                self.assertTrue(all("bbox_xyxy" not in query for query in records["queries"]))
                self.assertEqual(
                    sum(query["has_ground_truth"] for query in records["queries"]), 1
                )
                self.assertEqual(len(evaluation["classes"][class_name]), 1)
                self.assertTrue(Path(records["supports"][0]["image_path"]).is_file())
                self.assertIn("images", Path(records["supports"][0]["image_path"]).parts)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "evaluation.json").is_file())

    def test_external_unlabeled_split_creates_qualitative_queries_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            self._write_split(
                root,
                "set1",
                [1],
                [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 8, 7]}],
            )
            self._write_split(root, "external", [21, 22], [])
            manifest, evaluation = prepare_manifest(
                root,
                Path(directory) / "episodes",
                support_split="set1",
                query_split="external",
                class_names=("NT",),
            )
            queries = manifest["classes"]["NT"]["queries"]
            self.assertEqual(len(queries), 2)
            self.assertTrue(all(not item["has_ground_truth"] for item in queries))
            self.assertEqual(evaluation["classes"]["NT"], {})

    def test_bbox_and_box_proxy_metrics(self) -> None:
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:8, 3:9] = True
        self.assertEqual(bbox_iou([3, 2, 9, 8], [3, 2, 9, 8]), 1.0)
        metrics = box_proxy_mask_metrics(mask, [3, 2, 9, 8])
        self.assertEqual(metrics["box_proxy_mask_iou"], 1.0)
        self.assertEqual(metrics["box_proxy_dice"], 1.0)
        self.assertEqual(metrics["outside_box_fraction"], 0.0)

    def test_average_precision_perfect_prediction(self) -> None:
        targets = {"q1": [1, 1, 9, 9], "q2": [2, 2, 8, 8]}
        predictions = [
            {"query_id": "q1", "bbox_xyxy": targets["q1"], "score": 0.9},
            {"query_id": "q2", "bbox_xyxy": targets["q2"], "score": 0.8},
        ]
        self.assertEqual(average_precision_at_iou(predictions, targets, 0.5), 1.0)

    def test_average_precision_counts_class_negative_query_false_positives(self) -> None:
        targets = {"positive": [1, 1, 9, 9]}
        predictions = [
            {"query_id": "negative", "bbox_xyxy": [1, 1, 9, 9], "score": 0.95},
            {"query_id": "positive", "bbox_xyxy": [1, 1, 9, 9], "score": 0.80},
        ]
        self.assertLess(average_precision_at_iou(predictions, targets, 0.5), 1.0)

    def test_inference_uses_only_support_box_on_pair_canvas(self) -> None:
        class FakeProcessor:
            def __init__(self) -> None:
                self.image: Image.Image | None = None
                self.text: str | None = None
                self.boxes: list[tuple[list[float], bool]] = []

            def set_image(self, image: Image.Image) -> dict[str, Any]:
                self.image = image
                return {}

            def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
                self.text = prompt
                return state

            def add_geometric_prompt(
                self, box: list[float], label: bool, state: dict[str, Any]
            ) -> dict[str, Any]:
                self.boxes.append((box, label))
                assert self.image is not None
                mask = torch.zeros((1, 1, self.image.height, self.image.width), dtype=torch.bool)
                mask[:, :, :, self.image.width // 2 :] = True
                state["masks"] = mask
                state["scores"] = torch.tensor([0.9])
                return state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support_path = root / "support.png"
            query_path = root / "query.png"
            Image.new("RGB", (20, 10), color="red").save(support_path)
            Image.new("RGB", (12, 12), color="blue").save(query_path)
            support = [{"image_path": str(support_path), "bbox_xyxy": [2, 2, 12, 8]}]
            query = {"query_id": "q", "image_id": 2, "image_path": str(query_path)}
            processor = FakeProcessor()
            prediction, mask, composite, _ = _predict_one(
                processor, "NT", support, query, 32, ("pixels", 2.0)
            )

        self.assertEqual(composite.size, (32, 32))
        self.assertEqual(processor.text, "nuchal translucency (NT)")
        self.assertEqual(len(processor.boxes), 1)
        self.assertTrue(processor.boxes[0][1])
        self.assertEqual(prediction["support_prompt_boxes_xyxy"], [[0.0, 0.0, 14.0, 10.0]])
        self.assertNotIn("bbox_xyxy", query)
        self.assertEqual(mask.shape, (12, 12))
        self.assertTrue(prediction["mask_found"])


if __name__ == "__main__":
    unittest.main()