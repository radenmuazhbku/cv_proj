"""Box prompt tokenization for the RF-DETR PCS fork."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


class BoxPromptEncoder(nn.Module):
    """Encode multiple normalized ``cxcywh`` box prompts as a token sequence.

    Each token combines pooled appearance, box geometry, and a binary prompt
    label. Prompt tokens first exchange information with each other, then attend
    to the image feature map. No query annotations are accepted here.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, roi_size: int = 3) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.box_projection = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.roi_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.label_embedding = nn.Embedding(2, hidden_dim)
        self.prompt_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.image_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        labels: Tensor,
        image_size: tuple[int, int],
        box_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if feature_map.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("Expected feature_map [B,C,Hf,Wf], boxes [B,P,4].")
        batch, channels, feat_h, feat_w = feature_map.shape
        if boxes.shape[0] != batch or labels.shape != boxes.shape[:2]:
            raise ValueError("Prompt boxes and labels must have matching [B,P] dimensions.")
        if box_padding_mask is not None and box_padding_mask.shape != labels.shape:
            raise ValueError("box_padding_mask must have shape [B,P].")

        boxes = boxes.to(device=feature_map.device, dtype=feature_map.dtype)
        labels = labels.to(device=feature_map.device, dtype=torch.long).clamp(0, 1)
        image_h, image_w = image_size
        xyxy = torch.stack(
            (boxes[..., 0] - boxes[..., 2] / 2, boxes[..., 1] - boxes[..., 3] / 2,
             boxes[..., 0] + boxes[..., 2] / 2, boxes[..., 1] + boxes[..., 3] / 2), dim=-1
        )
        rois = torch.cat(
            (
                torch.arange(batch, device=boxes.device, dtype=boxes.dtype)
                .repeat_interleave(boxes.shape[1])[:, None],
                xyxy.reshape(-1, 4) * boxes.new_tensor([image_w, image_h, image_w, image_h]),
            ),
            dim=-1,
        )
        pooled = roi_align(
            feature_map, rois, output_size=(self.roi_size, self.roi_size),
            spatial_scale=((feat_w / image_w) + (feat_h / image_h)) / 2, aligned=True,
        ).mean(dim=(-1, -2)).reshape(batch, boxes.shape[1], channels)
        tokens = self.roi_projection(pooled) + self.box_projection(boxes) + self.label_embedding(labels)

        self_update = self.prompt_self_attn(
            tokens, tokens, tokens, key_padding_mask=box_padding_mask, need_weights=False
        )[0]
        tokens = self.norm1(tokens + self_update)
        image_memory = feature_map.flatten(2).transpose(1, 2)
        image_update = self.image_cross_attn(
            tokens, image_memory, image_memory, need_weights=False
        )[0]
        tokens = self.norm2(tokens + image_update)
        tokens = self.norm3(tokens + self.ffn(tokens))
        return tokens, box_padding_mask
