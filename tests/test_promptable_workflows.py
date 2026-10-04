from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from cv_proj.configuration import parse_args_with_python_config
from rfdetr_promptable.coco_episodes import CocoFewShotEpisodeDataset, collate_few_shot
from scripts.generate_coco_prompt_split import split_support_query_coco
from scripts.train_promptable_fewshot import run_coco_map_eval


class _PerfectPresenceDetector:
    """Small deterministic detector fixture for the COCOeval contract."""

    def __init__(self) -> None:
        self.training = True
        self.encode_calls = 0
        self.forward_calls = 0

    def eval(self) -> _PerfectPresenceDetector:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _PerfectPresenceDetector:
        self.training = mode
        return self

    def encode_support(
        self,
        support_images: torch.Tensor,
        support_boxes: torch.Tensor,
        support_group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encode_calls += 1
        assert support_group_ids is not None
        return torch.zeros((1, 4)), torch.unique(support_group_ids)

    def forward_support_query(
        self,
        support_images: torch.Tensor,
        support_boxes: torch.Tensor,
        query_images: torch.Tensor,
        support_group_ids: torch.Tensor | None = None,
        query_to_support: torch.Tensor | None = None,
        encoded_support: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        batch_size = query_images.shape[0]
        is_positive = query_images.mean(dim=(1, 2, 3)) > 0.5
        logits = torch.where(is_positive, 10.0, -10.0).view(batch_size, 1, 1)
        boxes = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 1, 4).expand(batch_size, 1, 4)
        return {"pred_logits": logits, "pred_boxes": boxes}


class PromptableWorkflowTests(unittest.TestCase):
    def _write_coco_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        image_root = root / "images"
        image_root.mkdir()
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        image_values = {10: 255, 11: 255, 20: 255, 21: 0}
        for image_id, value in image_values.items():
            name = f"{image_id}.png"
            Image.new("RGB", (32, 32), color=(value, value, value)).save(image_root / name)
            images.append({"id": image_id, "file_name": name, "width": 32, "height": 32})
            if image_id != 21:
                annotations.append(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [8, 8, 16, 16],
                        "area": 256,
                        "iscrowd": 0,
                    }
                )
        payload = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
        }
        support_path = root / "support.json"
        query_path = root / "query.json"
        support_path.write_text(json.dumps({**payload, "images": images[:2], "annotations": annotations[:2]}))
        query_path.write_text(json.dumps({**payload, "images": images[2:], "annotations": annotations[2:]}))
        return image_root, support_path, query_path

    def test_python_config_defaults_are_overridden_by_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.py"
            config_path.write_text(
                "from pathlib import Path\ncfg = {'steps': 12, 'output': Path('checkpoint.pt')}\n"
            )
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=Path)
            parser.add_argument("--steps", type=int, default=1)
            parser.add_argument("--output", type=Path)
            args = parse_args_with_python_config(
                parser, ["--config", str(config_path), "--steps", "3"]
            )
        self.assertEqual(args.steps, 3)
        self.assertEqual(args.output, Path("checkpoint.pt"))

    def test_support_query_split_is_image_disjoint(self) -> None:
        source = {
            "info": {},
            "licenses": [],
            "images": [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in range(1, 8)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 2, "category_id": 1},
                {"id": 3, "image_id": 3, "category_id": 1},
                {"id": 4, "image_id": 4, "category_id": 2},
                {"id": 5, "image_id": 5, "category_id": 2},
            ],
            "categories": [{"id": 1}, {"id": 2}],
        }
        support, query = split_support_query_coco(source, {1, 2}, 1, seed=4)
        support_ids = {image["id"] for image in support["images"]}
        query_ids = {image["id"] for image in query["images"]}
        self.assertEqual(support_ids | query_ids, set(range(1, 8)))
        self.assertFalse(support_ids & query_ids)
        self.assertEqual(len(support_ids), 2)

    def test_dataset_uses_separate_support_and_query_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_root, support_path, query_path = self._write_coco_fixture(Path(directory))
            dataset = CocoFewShotEpisodeDataset(
                image_root,
                query_path,
                resolution=32,
                max_shots=1,
                negative_eval_per_class=1,
                support_annotation_file=support_path,
                training=False,
            )
            batch = collate_few_shot([dataset[0], dataset[1]])
        for query_id, support_ids in zip(batch["query_image_ids"], batch["support_image_ids"], strict=True):
            self.assertNotIn(query_id, support_ids)
        self.assertEqual(batch["query_to_support"].tolist(), [0, 1])
        self.assertEqual(batch["query_boxes"].shape, (2, 1, 4))

    def test_coco_ap_uses_all_disjoint_queries_and_caches_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_root, support_path, query_path = self._write_coco_fixture(Path(directory))
            dataset = CocoFewShotEpisodeDataset(
                image_root,
                query_path,
                resolution=32,
                max_shots=1,
                negative_eval_per_class=0,
                support_annotation_file=support_path,
                training=False,
            )
            model = _PerfectPresenceDetector()
            metrics = run_coco_map_eval(
                model, dataset, torch.device("cpu"), max_shots=1, query_batch_size=1, seed=7
            )
        self.assertAlmostEqual(metrics["mAP"], 1.0, places=3)
        self.assertEqual(metrics["categories"][1]["query_images"], 2)
        self.assertEqual(model.encode_calls, 1)
        self.assertEqual(model.forward_calls, 2)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
