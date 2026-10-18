"""Prompt-conditioned detector entry points for RF-DETR PCS experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rfdetr_pcs.config import ModelConfig, TrainConfig
from rfdetr_pcs.assets.coco_classes import COCO_CLASSES
from rfdetr_pcs.models.lwdetr import LWDETR, build_model_from_config
from rfdetr_pcs.models.criterion import SetCriterion
from rfdetr_pcs.models.matcher import HungarianMatcher
from rfdetr_pcs.models.prompt_encoder import BoxPromptEncoder
from rfdetr_pcs.models.weights import load_pretrain_weights
from rfdetr_pcs.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list


class PCSDetector(nn.Module):
    """Box-prompt-conditioned RF-DETR with same-image and support/query APIs.

    Supports same-image prompts and a separate support/query extension. Both
    paths preserve the image batch axis and route box tokens into decoder layers.
    """

    def __init__(
        self, detector: LWDETR, prompt_heads: int = 8, num_prompt_classes: int = 91,
        init_class_embeddings_from_detector: bool = True, use_class_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if detector.group_detr != 1 or detector.two_stage:
            raise ValueError("PCSDetector currently requires group_detr=1 and two_stage=False.")
        self.detector = detector
        self.hidden_dim = detector.transformer.d_model
        self.prompt_encoder = BoxPromptEncoder(self.hidden_dim, num_heads=prompt_heads)
        # COCO category IDs index this table directly; slot zero is reserved.
        self.class_embeddings: nn.Embedding | None
        self.class_embeddings = nn.Embedding(num_prompt_classes, self.hidden_dim) if use_class_embeddings else None
        if self.class_embeddings is not None:
            nn.init.normal_(self.class_embeddings.weight, std=0.02)
        classifier = getattr(detector, "class_embed", None)
        if use_class_embeddings and init_class_embeddings_from_detector and isinstance(classifier, nn.Linear):
            # Match RF-DETR's actual pretrained COCO row order, which follows
            # COCO_CLASSES rather than category_id - 1 (COCO IDs contain gaps).
            # The vectors remain trainable and are optimized by PCS losses.
            with torch.no_grad():
                for row, category_id in enumerate(COCO_CLASSES):
                    if category_id < num_prompt_classes and row < classifier.out_features:
                        self.class_embeddings.weight[category_id].copy_(classifier.weight[row])
        # SAM3 scores each detection query against the encoded prompt. Keep the
        # prompt-conditioned dot-product path instead of a prompt-independent
        # objectness classifier.
        score_dim = self.hidden_dim
        self.query_score_projection = nn.Linear(self.hidden_dim, score_dim)
        self.prompt_score_projection = nn.Linear(self.hidden_dim, score_dim)
        self.score_scale = score_dim ** -0.5
        # A dedicated decoder query predicts whether the prompted concept is
        # present, following SAM3's presence token path.
        self.presence_query = nn.Parameter(torch.empty(1, self.hidden_dim))
        nn.init.normal_(self.presence_query, std=0.02)
        self.presence_reference = nn.Parameter(torch.zeros(1, 4))
        self.presence_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Linear(self.hidden_dim, 1),
        )
        self.matcher = HungarianMatcher(
            cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_mask_ce=1.0, cost_mask_dice=1.0
        )
        weights = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_presence": 1.0}
        losses = ["labels", "boxes"]
        if detector.segmentation_head is not None:
            weights.update({"loss_mask_ce": 5.0, "loss_mask_dice": 5.0})
            losses.append("masks")
        if detector.aux_loss:
            base_weights = dict(weights)
            for layer_index in range(detector.transformer.dec_layers - 1):
                weights.update({f"{name}_{layer_index}": value for name, value in base_weights.items()})
        self.criterion = SetCriterion(1, self.matcher, weights, focal_alpha=0.25, losses=losses)

    def _append_presence_query(self, refs: Tensor, queries: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.cat((refs, self.presence_reference.to(dtype=refs.dtype)), dim=0),
            torch.cat((queries, self.presence_query.to(dtype=queries.dtype)), dim=0),
        )

    def _append_class_token(
        self, tokens: Tensor, padding_mask: Tensor | None, class_ids: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if class_ids is None:
            return tokens, padding_mask
        if self.class_embeddings is None:
            raise ValueError("class_ids were provided, but this model was built without class embeddings.")
        class_ids = class_ids.to(device=tokens.device, dtype=torch.long).reshape(-1)
        if class_ids.shape[0] != tokens.shape[0]:
            raise ValueError("class_ids must have one COCO category ID per prompt sequence.")
        if (class_ids < 0).any() or (class_ids >= self.class_embeddings.num_embeddings).any():
            raise ValueError(f"class_ids must be in [0, {self.class_embeddings.num_embeddings - 1}].")
        class_token = self.class_embeddings(class_ids).unsqueeze(1).to(dtype=tokens.dtype)
        tokens = torch.cat((tokens, class_token), dim=1)
        if padding_mask is not None:
            class_padding = torch.zeros((len(class_ids), 1), dtype=torch.bool, device=tokens.device)
            padding_mask = torch.cat((padding_mask, class_padding), dim=1)
        return tokens, padding_mask

    def _score_queries(
        self, hs: Tensor, presence_state: Tensor, prompt_tokens: Tensor, padding_mask: Tensor | None
    ):
        """Return prompt similarity logits and an image-level presence logit.

        Args use decoder layout ``hs=[L,B,Q,D]`` and prompt layout ``[B,P,D]``.
        """
        if padding_mask is None:
            prompt = prompt_tokens.mean(dim=1)
        else:
            valid = (~padding_mask).to(prompt_tokens.dtype).unsqueeze(-1)
            prompt = (prompt_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        query_proj = self.query_score_projection(hs)
        prompt_proj = self.prompt_score_projection(prompt)[None, :, None, :]
        logits = ((query_proj * prompt_proj).sum(dim=-1, keepdim=True) * self.score_scale).clamp(-12.0, 12.0)
        # Presence comes from its own decoder token, separate from detection
        # query similarity as in SAM3.
        presence = self.presence_head(presence_state).squeeze(-1)
        return logits, presence

    def losses(
        self, outputs: dict[str, Any], targets: list[dict[str, Tensor]],
        presence_targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Detection/mask losses plus the prompted-class presence objective."""
        losses = self.criterion(outputs, targets)
        if presence_targets is None:
            presence_target = torch.tensor(
                [float(len(target["boxes"]) > 0) for target in targets],
                device=outputs["presence_logit_dec"].device,
                dtype=outputs["presence_logit_dec"].dtype,
            )
        else:
            presence_target = presence_targets.to(
                device=outputs["presence_logit_dec"].device,
                dtype=outputs["presence_logit_dec"].dtype,
            ).reshape_as(outputs["presence_logit_dec"])
        losses["loss_presence"] = F.binary_cross_entropy_with_logits(
            outputs["presence_logit_dec"], presence_target
        )
        for index, aux in enumerate(outputs.get("aux_outputs", [])):
            losses[f"loss_presence_{index}"] = F.binary_cross_entropy_with_logits(
                aux["presence_logit_dec"], presence_target
            )
        return losses

    @classmethod
    def from_config(
        cls, model_config: ModelConfig, train_config: TrainConfig | None = None, prompt_heads: int = 8,
        num_prompt_classes: int = 91, init_class_embeddings_from_detector: bool = True,
        use_class_embeddings: bool = True,
    ) -> "PCSDetector":
        if train_config is None:
            train_config = TrainConfig(dataset_dir=".", output_dir=".")
        detector = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            load_pretrain_weights(detector, model_config)
        return cls(
            detector, prompt_heads=prompt_heads,
            num_prompt_classes=num_prompt_classes,
            init_class_embeddings_from_detector=init_class_embeddings_from_detector,
            use_class_embeddings=use_class_embeddings,
        )

    @staticmethod
    def _extract_features(detector: LWDETR, images: Tensor):
        nested = nested_tensor_from_tensor_list(images)
        features, positions, cross_features = detector.backbone(nested)
        srcs, masks = [], []
        for feature in features:
            src, mask = feature.decompose()
            srcs.append(src)
            masks.append(mask)
        cross_srcs = None if cross_features is None else [feature.decompose()[0] for feature in cross_features]
        return nested, features, positions, srcs, masks, cross_srcs

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        """Load matching weights while leaving newly added PCS heads initialized.

        This supports warm-starting pre-class-prompt PCS checkpoints. Shape-mismatched
        classifier heads are skipped explicitly; all omitted keys are returned so
        callers can report the migration.
        """
        current = self.state_dict()
        compatible = {
            key: value for key, value in state_dict.items()
            if key in current and torch.is_tensor(value) and current[key].shape == value.shape
        }
        skipped = [key for key in state_dict if key not in compatible]
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return sorted(set(missing)), sorted(set(skipped) | set(unexpected))

    def forward(
        self,
        query_images: Tensor,
        prompt_boxes: Tensor,
        prompt_labels: Tensor | None = None,
        prompt_padding_mask: Tensor | None = None,
        class_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Same-image PCS: prompt boxes and detections refer to ``query_images``.

        ``prompt_boxes`` uses normalized ``cxcywh`` coordinates. ``prompt_labels``
        is 1 for positive examples and 0 for negative examples. Ground-truth
        detections are deliberately not an input to this function.
        """
        if query_images.ndim != 4:
            raise ValueError("query_images must have shape [B,3,H,W].")
        if prompt_boxes.ndim == 2:
            prompt_boxes = prompt_boxes[:, None]
        if prompt_boxes.ndim != 3 or prompt_boxes.shape[0] != query_images.shape[0] or prompt_boxes.shape[-1] != 4:
            raise ValueError("prompt_boxes must have shape [B,P,4] or [B,4].")
        if prompt_labels is None:
            prompt_labels = torch.ones(prompt_boxes.shape[:2], dtype=torch.long, device=prompt_boxes.device)

        nested, features, positions, srcs, masks, cross_srcs = self._extract_features(self.detector, query_images)
        prompt_tokens, prompt_padding_mask = self.prompt_encoder(
            features[-1].tensors,
            prompt_boxes,
            prompt_labels,
            tuple(query_images.shape[-2:]),
            prompt_padding_mask,
        )
        prompt_tokens, prompt_padding_mask = self._append_class_token(
            prompt_tokens, prompt_padding_mask, class_ids
        )
        if self.training:
            refs = self.detector.refpoint_embed.weight
            queries = self.detector.query_feat.weight
        else:
            refs = self.detector.refpoint_embed.weight[: self.detector.num_queries]
            queries = self.detector.query_feat.weight[: self.detector.num_queries]
        refs, queries = self._append_presence_query(refs, queries)
        transformed = self.detector.transformer(
            srcs,
            masks,
            positions,
            refs,
            queries,
            cross_attn_srcs=cross_srcs,
            prompt_tokens=prompt_tokens,
            prompt_padding_mask=prompt_padding_mask,
        )
        hs, references = transformed[:2]
        if hs is None or references is None:
            raise RuntimeError("PCS requires decoder outputs; configure dec_layers > 0.")
        presence_state = hs[:, :, -1]
        hs, references = hs[:, :, :-1], references[:, :, :-1]

        if self.detector.bbox_reparam:
            delta = self.detector.bbox_embed(hs)
            boxes = torch.cat(
                [delta[..., :2] * references[..., 2:] + references[..., :2],
                 delta[..., 2:].exp() * references[..., 2:]], dim=-1
            )
        else:
            boxes = (self.detector.bbox_embed(hs) + references).sigmoid()
        logits, presence = self._score_queries(hs, presence_state, prompt_tokens, prompt_padding_mask)
        out: dict[str, Any] = {
            "pred_logits": logits[-1], "pred_boxes": boxes[-1], "presence_logit_dec": presence[-1],
        }
        if class_ids is not None:
            out["prompt_class_ids"] = class_ids.to(device=query_images.device, dtype=torch.long).reshape(-1)

        if self.detector.segmentation_head is not None:
            mask_layers = self.detector.segmentation_head.sparse_forward(
                features[0].tensors, hs, tuple(nested.tensors.shape[-2:])
            )
            out["pred_masks"] = mask_layers[-1]
        if self.detector.aux_loss:
            aux = []
            for layer in range(logits.shape[0] - 1):
                item: dict[str, Any] = {
                    "pred_logits": logits[layer], "pred_boxes": boxes[layer],
                    "presence_logit_dec": presence[layer],
                }
                if self.detector.segmentation_head is not None:
                    item["pred_masks"] = mask_layers[layer]
                aux.append(item)
            out["aux_outputs"] = aux
        return out

    def forward_support_query(
        self,
        support_images: Tensor,
        support_boxes: Tensor,
        query_images: Tensor,
        support_labels: Tensor | None = None,
        support_padding_mask: Tensor | None = None,
        return_support_outputs: bool = False,
        class_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Condition each query image on one or more separately encoded supports.

        Args:
            support_images: ``[B,S,3,Hs,Ws]`` episode support images.
            support_boxes: ``[B,S,P,4]`` normalized ``cxcywh`` prompts.
            query_images: ``[B,Q,3,Hq,Wq]`` query images. Supports and queries
                need not be the same images or have the same spatial dimensions.
            support_labels: optional ``[B,S,P]`` positive/negative prompt labels.
            support_padding_mask: optional ``[B,S,P]`` mask where True means
                the prompt slot is padding.

        Support and query images are concatenated only on the image batch axis
        for one backbone call, then split back into their feature maps. Spatial
        feature maps are never tiled together. The encoded support box tokens
        are concatenated per episode, then broadcast over that episode's
        queries. Decoder prompt attention routes them to each query's object
        queries. Outputs are flattened in B-major, Q-minor order so they can be
        passed directly to the existing criterion.
        """
        if support_images.ndim != 5 or query_images.ndim != 5:
            raise ValueError("support_images/query_images must have shapes [B,S,3,H,W] and [B,Q,3,H,W].")
        episode_count, support_count = support_images.shape[:2]
        query_episode_count, query_count = query_images.shape[:2]
        if episode_count != query_episode_count or support_count < 1 or query_count < 1:
            raise ValueError("Support and query episode dimensions must match and contain images.")
        if support_boxes.ndim != 4 or support_boxes.shape[:2] != (episode_count, support_count) or support_boxes.shape[-1] != 4:
            raise ValueError("support_boxes must have shape [B,S,P,4].")
        if support_labels is None:
            support_labels = torch.ones(support_boxes.shape[:-1], dtype=torch.long, device=support_boxes.device)
        if support_padding_mask is not None:
            if support_padding_mask.shape != support_boxes.shape[:-1]:
                raise ValueError("support_padding_mask must have shape [B,S,P].")
            if support_padding_mask.all(dim=-1).any():
                raise ValueError("Each support image must have at least one non-padding prompt.")

        flat_support_images = support_images.reshape(episode_count * support_count, *support_images.shape[2:])
        flat_queries = query_images.reshape(episode_count * query_count, *query_images.shape[2:])
        if flat_support_images.shape[-2:] != flat_queries.shape[-2:]:
            raise ValueError("Batch-dimension support/query encoding requires equal padded image sizes.")
        support_batch_size = flat_support_images.shape[0]
        all_images = torch.cat((flat_support_images, flat_queries), dim=0)
        nested, all_features, all_positions, all_srcs, all_masks, all_cross_srcs = self._extract_features(
            self.detector, all_images
        )
        flat_boxes = support_boxes.reshape(episode_count * support_count, support_boxes.shape[2], 4)
        flat_labels = support_labels.reshape(episode_count * support_count, support_labels.shape[2])
        flat_padding = None if support_padding_mask is None else support_padding_mask.reshape(
            episode_count * support_count, support_padding_mask.shape[2]
        )
        support_tokens, flat_padding = self.prompt_encoder(
            all_features[-1].tensors[:support_batch_size],
            flat_boxes,
            flat_labels,
            tuple(flat_support_images.shape[-2:]),
            flat_padding,
        )
        prompt_count = support_tokens.shape[1]
        flat_support_tokens, flat_padding = self._append_class_token(
            support_tokens,
            flat_padding,
            None if class_ids is None else class_ids.to(support_tokens.device).repeat_interleave(support_count),
        )
        # Keep one prompt sequence per support image for optional auxiliary
        # support decoding. The episode-shaped version below is only for routing
        # the combined support tokens to query images.
        support_tokens = support_tokens.reshape(
            episode_count, support_count * prompt_count, self.hidden_dim
        )
        if flat_padding is None:
            episode_padding = None
        else:
            episode_padding = flat_padding[:, :prompt_count].reshape(
                episode_count, support_count * prompt_count
            )
        support_tokens, episode_padding = self._append_class_token(
            support_tokens, episode_padding, class_ids
        )
        routed_prompt_count = support_tokens.shape[1]
        # Route each episode's support prompts to all its queries in B-major order.
        routed_tokens = support_tokens[:, None].expand(-1, query_count, -1, -1).reshape(
            episode_count * query_count, routed_prompt_count, self.hidden_dim
        )
        routed_padding = None if episode_padding is None else episode_padding[:, None].expand(
            -1, query_count, -1
        ).reshape(episode_count * query_count, routed_prompt_count)

        query_features = [
            NestedTensor(item.tensors[support_batch_size:],
                         None if item.mask is None else item.mask[support_batch_size:])
            for item in all_features
        ]
        query_positions = [
            position if position.shape[0] == 1 else position[support_batch_size:]
            for position in all_positions
        ]
        query_srcs = [source[support_batch_size:] for source in all_srcs]
        query_masks = [mask[support_batch_size:] for mask in all_masks]
        query_cross_srcs = None if all_cross_srcs is None else [
            source[support_batch_size:] for source in all_cross_srcs
        ]
        query_nested = NestedTensor(
            nested.tensors[support_batch_size:],
            None if nested.mask is None else nested.mask[support_batch_size:],
        )
        if self.training:
            refs, queries = self.detector.refpoint_embed.weight, self.detector.query_feat.weight
        else:
            refs = self.detector.refpoint_embed.weight[: self.detector.num_queries]
            queries = self.detector.query_feat.weight[: self.detector.num_queries]
        refs, queries = self._append_presence_query(refs, queries)
        transformed = self.detector.transformer(
            query_srcs, query_masks, query_positions, refs, queries,
            cross_attn_srcs=query_cross_srcs,
            prompt_tokens=routed_tokens,
            prompt_padding_mask=routed_padding,
        )
        hs, references = transformed[:2]
        if hs is None or references is None:
            raise RuntimeError("PCS requires decoder outputs; configure dec_layers > 0.")
        presence_state = hs[:, :, -1]
        hs, references = hs[:, :, :-1], references[:, :, :-1]
        if self.detector.bbox_reparam:
            delta = self.detector.bbox_embed(hs)
            boxes = torch.cat(
                [delta[..., :2] * references[..., 2:] + references[..., :2],
                 delta[..., 2:].exp() * references[..., 2:]], dim=-1
            )
        else:
            boxes = (self.detector.bbox_embed(hs) + references).sigmoid()
        logits, presence = self._score_queries(hs, presence_state, routed_tokens, routed_padding)
        output: dict[str, Any] = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "presence_logit_dec": presence[-1],
            "episode_shape": (episode_count, query_count),
        }
        if class_ids is not None:
            output["prompt_class_ids"] = class_ids.to(device=query_images.device, dtype=torch.long).repeat_interleave(
                query_count
            )
        if self.detector.segmentation_head is not None:
            mask_layers = self.detector.segmentation_head.sparse_forward(
                query_features[0].tensors, hs, tuple(query_nested.tensors.shape[-2:])
            )
            output["pred_masks"] = mask_layers[-1]
        if self.detector.aux_loss:
            aux = []
            for layer in range(logits.shape[0] - 1):
                item: dict[str, Any] = {
                    "pred_logits": logits[layer], "pred_boxes": boxes[layer],
                    "presence_logit_dec": presence[layer],
                }
                if self.detector.segmentation_head is not None:
                    item["pred_masks"] = mask_layers[layer]
                aux.append(item)
            output["aux_outputs"] = aux
        if return_support_outputs:
            support_positions = [
                position if position.shape[0] == 1 else position[:support_batch_size]
                for position in all_positions
            ]
            support_srcs = [source[:support_batch_size] for source in all_srcs]
            support_masks = [mask[:support_batch_size] for mask in all_masks]
            support_cross_srcs = None if all_cross_srcs is None else [
                source[:support_batch_size] for source in all_cross_srcs
            ]
            if self.training:
                support_refs, support_queries = self.detector.refpoint_embed.weight, self.detector.query_feat.weight
            else:
                support_refs = self.detector.refpoint_embed.weight[: self.detector.num_queries]
                support_queries = self.detector.query_feat.weight[: self.detector.num_queries]
            support_refs, support_queries = self._append_presence_query(support_refs, support_queries)
            support_transformed = self.detector.transformer(
                support_srcs, support_masks, support_positions, support_refs, support_queries,
                cross_attn_srcs=support_cross_srcs,
                prompt_tokens=flat_support_tokens,
                prompt_padding_mask=flat_padding,
            )
            support_hs, support_references = support_transformed[:2]
            if support_hs is None or support_references is None:
                raise RuntimeError("PCS requires decoder outputs; configure dec_layers > 0.")
            support_presence_state = support_hs[:, :, -1]
            support_hs, support_references = support_hs[:, :, :-1], support_references[:, :, :-1]
            if self.detector.bbox_reparam:
                support_delta = self.detector.bbox_embed(support_hs)
                support_boxes = torch.cat(
                    [support_delta[..., :2] * support_references[..., 2:] + support_references[..., :2],
                     support_delta[..., 2:].exp() * support_references[..., 2:]], dim=-1
                )
            else:
                support_boxes = (self.detector.bbox_embed(support_hs) + support_references).sigmoid()
            support_logits, support_presence = self._score_queries(
                support_hs, support_presence_state, flat_support_tokens, flat_padding
            )
            support_output: dict[str, Any] = {
                "pred_logits": support_logits[-1], "pred_boxes": support_boxes[-1],
                "presence_logit_dec": support_presence[-1],
            }
            if self.detector.segmentation_head is not None:
                support_mask_layers = self.detector.segmentation_head.sparse_forward(
                    all_features[0].tensors[:support_batch_size], support_hs,
                    tuple(nested.tensors.shape[-2:]),
                )
                support_output["pred_masks"] = support_mask_layers[-1]
            if self.detector.aux_loss:
                support_aux = []
                for layer in range(support_logits.shape[0] - 1):
                    item: dict[str, Any] = {
                        "pred_logits": support_logits[layer], "pred_boxes": support_boxes[layer],
                        "presence_logit_dec": support_presence[layer],
                    }
                    if self.detector.segmentation_head is not None:
                        item["pred_masks"] = support_mask_layers[layer]
                    support_aux.append(item)
                support_output["aux_outputs"] = support_aux
            output["support_outputs"] = support_output
        return output

    def forward_same_image(self, images: Tensor, prompt_boxes: Tensor, prompt_labels: Tensor | None = None):
        """Named alias used by the episodic train/evaluation utilities."""
        return self.forward(images, prompt_boxes, prompt_labels)

    def compute_losses(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        *,
        include_prompt_box: bool = True,
    ) -> dict[str, Tensor]:
        """Build query-only targets and compute RF-DETR Hungarian detection/mask losses."""
        boxes = batch["query_boxes"].to(outputs["pred_boxes"].device)
        valid = batch["query_target_mask"].to(device=boxes.device, dtype=torch.bool)
        masks = batch.get("query_masks")
        if masks is not None:
            masks = masks.to(device=boxes.device)
        if include_prompt_box:
            prompt_boxes = batch["support_boxes"].to(device=boxes.device)
            prompt_valid = batch.get("prompt_target_mask")
            if prompt_valid is None:
                prompt_valid = batch["present"].reshape(-1, 1).expand(prompt_boxes.shape[:2]) > 0.5
            else:
                prompt_valid = prompt_valid.to(device=boxes.device, dtype=torch.bool)
            padding = batch.get("prompt_padding_mask")
            if padding is not None:
                prompt_valid &= ~padding.to(device=boxes.device, dtype=torch.bool)
            boxes = torch.cat([boxes, prompt_boxes], dim=1)
            valid = torch.cat([valid, prompt_valid], dim=1)
            if masks is not None:
                prompt_masks = batch["prompt_masks"].to(device=boxes.device)
                masks = torch.cat([masks, prompt_masks], dim=1)

        targets: list[dict[str, Tensor]] = []
        for index in range(boxes.shape[0]):
            target: dict[str, Tensor] = {
                "boxes": boxes[index, valid[index]],
                "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=boxes.device),
            }
            if masks is not None:
                target["masks"] = masks[index, valid[index]]
            targets.append(target)

        losses = self.losses(outputs, targets)
        weighted = {name: value * self.criterion.weight_dict[name] for name, value in losses.items()
                    if name in self.criterion.weight_dict}
        weighted["loss_total"] = sum(weighted.values())
        return weighted
