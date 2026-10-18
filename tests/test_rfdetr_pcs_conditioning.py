from types import SimpleNamespace

import torch
from torch import nn

from rfdetr_pcs.conditioning import class_ids_for_mode, conditioning_mode_at, uses_class_embeddings
from rfdetr_pcs.models.pcs import PCSDetector


class _DummyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_detr = 1
        self.two_stage = False
        self.transformer = SimpleNamespace(d_model=16, dec_layers=1)
        self.class_embed = nn.Linear(16, 90)
        self.segmentation_head = None
        self.aux_loss = False


def test_visual_prompt_has_no_class_table_or_class_token():
    model = PCSDetector(_DummyDetector(), use_class_embeddings=False)
    tokens = torch.randn(2, 3, 16)
    updated, padding = model._append_class_token(tokens, None, None)
    assert model.class_embeddings is None
    assert updated is tokens
    assert padding is None


def test_class_embedding_mode_appends_trainable_class_token():
    model = PCSDetector(_DummyDetector(), num_prompt_classes=91)
    tokens = torch.randn(2, 3, 16)
    updated, _ = model._append_class_token(tokens, None, torch.tensor([1, 90]))
    assert updated.shape == (2, 4, 16)
    updated.sum().backward()
    assert model.class_embeddings.weight.grad is not None


def test_curriculum_switches_from_class_embedding_to_visual_prompt():
    cfg = {
        "conditioning_mode": "visual_prompt",
        "conditioning_schedule": [
            {"mode": "class_embedding", "steps": 3},
            {"mode": "visual_prompt", "steps": 7},
        ],
    }
    assert uses_class_embeddings(cfg)
    assert conditioning_mode_at(cfg, 0) == "class_embedding"
    assert conditioning_mode_at(cfg, 2) == "class_embedding"
    assert conditioning_mode_at(cfg, 3) == "visual_prompt"
    batch = {"category_ids": torch.tensor([3, 4])}
    assert class_ids_for_mode(batch, "cpu", "visual_prompt") is None
    assert torch.equal(class_ids_for_mode(batch, "cpu", "class_embedding"), batch["category_ids"])

