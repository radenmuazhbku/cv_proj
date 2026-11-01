from __future__ import annotations

import unittest

import torch
from faster_coco_eval import COCO

from rfdetr.evaluation.coco_eval import CocoEvaluator


class KeypointEvaluationTests(unittest.TestCase):
    def test_skips_box_only_annotations_without_mutating_source(self) -> None:
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1, "width": 32, "height": 32}],
            "categories": [{"id": 1, "name": "thing", "keypoints": ["a", "b"], "skeleton": [[1, 2]]}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [2, 2, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                    "keypoints": [5, 5, 2, 15, 15, 2],
                    "num_keypoints": 2,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1, 1, 5, 5],
                    "area": 25,
                    "iscrowd": 0,
                    "keypoints": [],
                    "num_keypoints": 0,
                },
            ],
        }
        coco.createIndex()

        evaluator = CocoEvaluator(coco, ["keypoints"], keypoint_oks_sigmas=[0.1, 0.1], log_summary=False)
        evaluator.update(
            {
                1: {
                    "boxes": torch.tensor([[2.0, 2.0, 22.0, 22.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                    "keypoints": torch.tensor([[[5.0, 5.0, 0.9], [15.0, 15.0, 0.9]]]),
                }
            }
        )
        evaluator.synchronize_between_processes()
        evaluator.accumulate()

        self.assertEqual(len(coco.dataset["annotations"]), 2)
        self.assertEqual([annotation["id"] for annotation in evaluator.coco_gt.dataset["annotations"]], [1])
        self.assertEqual(evaluator.coco_eval["keypoints"].stats.shape, (10,))


if __name__ == "__main__":
    unittest.main()
