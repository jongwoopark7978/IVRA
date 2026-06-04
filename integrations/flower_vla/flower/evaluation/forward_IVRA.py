"""IVRA inference patch for FLOWER VLA.

This module monkey-patches FLOWER's observation encoder at evaluation time.
IVRA refines visual token relations by computing correlations between DaViT
key features and using them to pool visual tokens inside the language encoder.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)


def _get_known_qkv_projection(vision_tower: nn.Module) -> nn.Module:
    """Return the DaViT qkv projection used to build IVRA token correlations."""
    candidates = (
        ("last block channel attention", lambda m: m.blocks[-1][0][1].channel_attn.fn.qkv),
        ("last block window attention", lambda m: m.blocks[-1][0][0].window_attn.fn.qkv),
    )
    for name, getter in candidates:
        try:
            module = getter(vision_tower)
        except (AttributeError, IndexError, TypeError):
            continue
        logger.debug("Using DaViT qkv projection from %s.", name)
        return module
    raise RuntimeError(
        "Could not locate the DaViT qkv projection in FLOWER's vision tower. "
        "The IVRA patch expects the upstream FLOWER DaViT block layout."
    )


def _run_vision_tower(vision_tower: nn.Module, images: torch.Tensor):
    if hasattr(vision_tower, "forward_features_unpool"):
        return vision_tower.forward_features_unpool(images)
    return vision_tower(images)


def extract_last_qkv(vision_tower: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], int]:
    """Run the vision tower once and capture the selected qkv projection output."""
    qkv_projection = _get_known_qkv_projection(vision_tower)
    captured = {"qkv": None}

    def save_qkv(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        captured["qkv"] = output.detach()

    handle = qkv_projection.register_forward_hook(save_qkv)
    try:
        _run_vision_tower(vision_tower, images)
    finally:
        handle.remove()

    qkv = captured["qkv"]
    if qkv is None:
        raise RuntimeError("The DaViT qkv hook was not called during the vision forward pass.")

    if qkv.shape[-1] % 3 == 0:
        return qkv, torch.chunk(qkv, 3, dim=-1), -1
    if qkv.dim() >= 2 and qkv.shape[1] % 3 == 0:
        return qkv, torch.chunk(qkv, 3, dim=1), 1
    raise RuntimeError(f"Could not split qkv tensor into q/k/v parts: shape={tuple(qkv.shape)}")


def _key_features_to_tokens(keys: torch.Tensor, split_dim: int) -> torch.Tensor:
    """Convert captured key features into [batch, token, channel]."""
    if split_dim == -1:
        return keys.reshape(keys.shape[0], -1, keys.shape[-1])
    if split_dim == 1:
        return keys.flatten(2).transpose(1, 2)
    raise ValueError(f"Unsupported qkv split dimension: {split_dim}")


def _block_diag_time_correlations(
    correlations: torch.Tensor,
    batch_size: int,
    time_steps: int,
) -> torch.Tensor:
    """Keep visual correlations local to each image when FLOWER receives T > 1 frames."""
    if time_steps == 1:
        return correlations.reshape(batch_size, correlations.shape[1], 1, correlations.shape[-1])

    num_tokens = correlations.shape[1]
    correlations = correlations.reshape(batch_size, time_steps, num_tokens, 1, num_tokens)
    merged = correlations.new_zeros(batch_size, time_steps * num_tokens, 1, time_steps * num_tokens)
    for t in range(time_steps):
        start = t * num_tokens
        end = start + num_tokens
        merged[:, start:end, :, start:end] = correlations[:, t]
    return merged


@torch.no_grad()
def _ivra_get_corrs(
    self,
    images: torch.Tensor,
    batch_size: Optional[int] = None,
    time_steps: int = 1,
) -> torch.Tensor:
    """Compute visual-token correlations from the selected DaViT key features."""
    _, (_, keys, _), split_dim = extract_last_qkv(self.vlm.vision_tower, images)
    key_tokens = _key_features_to_tokens(keys.float(), split_dim)
    key_tokens = F.normalize(key_tokens, dim=-1, eps=1e-6)
    correlations = torch.bmm(key_tokens, key_tokens.transpose(1, 2)).unsqueeze(2)

    if batch_size is None:
        return correlations
    return _block_diag_time_correlations(correlations, batch_size, time_steps)


def compute_weighted_pool(token_features: torch.Tensor, correlations: torch.Tensor) -> torch.Tensor:
    """Pool visual tokens with IVRA correlations and return [batch, token, channel]."""
    if token_features.dim() != 3:
        raise ValueError(f"Expected token features with shape [B, N, C], got {tuple(token_features.shape)}")

    token_grid = token_features.transpose(1, 2).unsqueeze(2)
    if token_grid.shape[-2:] != correlations.shape[-2:]:
        token_grid = F.interpolate(
            token_grid,
            size=correlations.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    weights = correlations.to(dtype=token_grid.dtype, device=token_grid.device)
    pooled = torch.einsum("bnij,bcij->bcn", weights, token_grid)
    norm = weights.flatten(-2, -1).sum(dim=-1).clamp_min(1e-6).unsqueeze(1)
    return (pooled / norm).transpose(1, 2)


def _prepare_attention_mask(model, attention_mask, inputs_embeds, head_mask, output_attentions):
    if attention_mask is None:
        return None
    if getattr(model, "_use_flash_attention_2", False):
        return attention_mask if (attention_mask == 0).any() else None
    if getattr(model, "_use_sdpa", False) and head_mask is None and not output_attentions:
        return _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
    return _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)


def _apply_ivra_to_span(
    hidden_states: torch.Tensor,
    start: int,
    correlations: Optional[torch.Tensor],
    weight: float,
) -> torch.Tensor:
    if correlations is None:
        return hidden_states

    num_tokens = correlations.shape[1]
    end = start + num_tokens
    if end > hidden_states.shape[1]:
        raise RuntimeError(
            f"IVRA visual-token span [{start}, {end}) exceeds encoder sequence length "
            f"{hidden_states.shape[1]}."
        )

    updated = hidden_states.clone()
    pooled = compute_weighted_pool(hidden_states[:, start:end], correlations)
    updated[:, start:end] = (1.0 - weight) * hidden_states[:, start:end] + weight * pooled
    return updated


def forward_weighting(
    model,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    mask: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask_idx: int = -1,
    ivra_weight: float = 0.2,
) -> Union[Tuple, BaseModelOutput]:
    """FLOWER encoder forward with one IVRA visual-token refinement layer."""
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Specify either input_ids or inputs_embeds, not both.")
    if input_ids is not None:
        input_shape_source = input_ids
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        inputs_embeds = model.embed_tokens(input_ids)
    elif inputs_embeds is not None:
        input_shape_source = inputs_embeds[:, :, -1]
    else:
        raise ValueError("Specify input_ids or inputs_embeds.")

    embed_pos = model.embed_positions(input_shape_source).to(inputs_embeds.device)
    hidden_states = model.layernorm_embedding(inputs_embeds + embed_pos)
    hidden_states = nn.functional.dropout(hidden_states, p=model.dropout, training=model.training)
    attention_mask = _prepare_attention_mask(model, attention_mask, inputs_embeds, head_mask, output_attentions)

    if head_mask is not None and head_mask.size()[0] != len(model.layers):
        raise ValueError(f"head_mask has {head_mask.size()[0]} layers, expected {len(model.layers)}.")

    target_idx = mask_idx if mask_idx >= 0 else len(model.layers) + mask_idx
    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for idx, encoder_layer in enumerate(model.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        layer_outputs = encoder_layer(
            hidden_states,
            attention_mask,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if idx == target_idx and mask is not None:
            hidden_states = _apply_ivra_to_span(hidden_states, 0, mask, ivra_weight)
            hidden_states = _apply_ivra_to_span(hidden_states, mask.shape[1], mask2, ivra_weight)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in (hidden_states, encoder_states, all_attentions) if v is not None)
    return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


def _ivra_encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
    """Encode FLOWER observations with IVRA visual-token refinement."""
    device = self.device
    default_type = next(self.parameters()).dtype

    image_tensor = batch["rgb_obs"]["rgb_static"]
    batch_size, time_steps, channels, height, width = image_tensor.shape
    flat_images = image_tensor.view(-1, channels, height, width).to(device=device, dtype=default_type)

    image_features = self.vlm._encode_image(flat_images).to(default_type)
    image_features = image_features.view(batch_size, time_steps * image_features.shape[1], -1)
    masks = self.get_corrs(flat_images, batch_size=batch_size, time_steps=time_steps)

    masks2 = None
    if self.use_second_view:
        image2_tensor = batch["rgb_obs"]["rgb_gripper"]
        flat_images2 = image2_tensor.view(-1, channels, height, width).to(device=device, dtype=default_type)
        image2_features = self.vlm._encode_image(flat_images2).to(default_type)
        image2_features = image2_features.view(batch_size, time_steps * image2_features.shape[1], -1)
        masks2 = self.get_corrs(flat_images2, batch_size=batch_size, time_steps=time_steps)
        image_features = torch.cat([image_features, image2_features], dim=1)

    constructed_prompts = self.construct_prompts(batch)
    text_embeds = self._get_text_embeddings(constructed_prompts, device)
    task_prompt = self.prompt_embeds.expand(batch_size, -1, -1).to(image_features.device)

    merged_embeds = torch.cat(
        [image_features, task_prompt, text_embeds.to(image_features.device)],
        dim=1,
    )
    attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
    layer_index = getattr(self, "_ivra_layer_index", -1)
    weight = getattr(self, "_ivra_weight", 0.2)

    features = forward_weighting(
        self.vlm.get_encoder(),
        inputs_embeds=merged_embeds,
        attention_mask=attention_mask,
        mask=masks,
        mask2=masks2,
        mask_idx=layer_index,
        ivra_weight=weight,
    ).last_hidden_state
    features = self.vlm_token_dropout(features)

    frequency_ids = torch.full((batch_size, 1, 1), 3, device=device, dtype=default_type)
    action_type = torch.ones(batch_size, self.act_window_size, 7, device=device, dtype=default_type)

    proprio = None
    if getattr(self, "use_proprio", False):
        obs_modalities = getattr(self, "obs_modalities", None)
        obs_source = batch.get(obs_modalities, {}) if obs_modalities is not None else {}
        if isinstance(obs_source, dict) and "proprio" in obs_source:
            proprio = obs_source["proprio"].to(device=device, dtype=default_type)

    return {
        "features": features,
        "frequency_embeds": self.frequency_embedder(frequency_ids),
        "action_space_embeds": None,
        "action_type": action_type,
        "proprio": proprio,
        "attention_mask": attention_mask,
    }


def patch_encode_observations(model, weight: float = 0.2, layer_index: int = -1):
    """Patch a FLOWER policy so inference uses IVRA-enhanced visual tokens.

    Args:
        model: FLOWER policy instance.
        weight: Blend weight for IVRA-pooled visual tokens.
        layer_index: Encoder layer where IVRA refinement is applied. ``-1`` means
            the final encoder layer.
    """
    model._ivra_weight = float(weight)
    model._ivra_layer_index = int(layer_index)
    model.encode_observations = MethodType(_ivra_encode_observations, model)
    model.get_corrs = MethodType(_ivra_get_corrs, model)
    return model
