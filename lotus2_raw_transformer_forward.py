"""ComfyUI node: Single-step raw Flux transformer forward pass on packed latents.

Step 6 of todo.md — executes one denoising step through the FLUX transformer
with the currently active PEFT adapter (core_predictor or detail_sharpener).
Accepts packed-sequence latents [B, T_seq, 4C] and returns the transformed
packed output for downstream unpacking + decoding.

Pipeline origin (from Lotus-2/pipeline.py):
    - Lines ~137–152: core_predictor block — single forward pass at fixed timestep,
      followed by _unpack_latents() → local_continuity_module().
    - The detail_sharpener path re-packs LCM output then enters a denoising loop,
      calling self.transformer(...) per-step with the same signature.

This node isolates that single ``self.transformer(...)`  call so the caller can
drive custom sigma scheduling in a separate sampler node (Step 7).
"""

import logging
from typing import Tuple

import torch

# Type-hint only — Lotus2ModelState lives in lotus2_peft_loader.

try:
    from .lotus2_peft_loader import Lotus2ModelState
except ImportError:
    # Allow instantiation without full ComfyUI env (e.g. unit tests).
    Lotus2ModelState = None  # type: ignore

from .lotus2_adapter_switcher import Lotus2AdapterSwitcher


logger = logging.getLogger(__name__)


class Lotus2RawTransformerForward:
    """Run a single Flux transformer forward pass on packed latents.

    Mirrors the ``self.transformer(...)`` call inside pipeline.py lines ~137–152.
    The active adapter must be set beforehand via Lotus2AdapterSwitcher.

    Input shape expectation for *packed_latents["samples"]*:
        [B, T_seq, 4C] where C = latent_channels (typically 64).

    Returns:
        LATENT dict containing the transformed packed tensor (same shape as input).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_model": ("LOTUS_MODEL",),
                "packed_latents": ("LATENT",),
                "timestep": ("FLOAT", {"default": 0.001}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
            },
            "optional": {
                "prompt_embeds": ("CONDITIONING",),
                "pooled_embeds": ("STRING", {"default": ""}),
                "img_ids": ("IMG_IDS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("packed_output_latents",)
    FUNCTION = "forward_pass"
    CATEGORY = "image/lotus2_decomposed"

    def forward_pass(
        self,
        lotus_model: "Lotus2ModelState",
        packed_latents: dict,                   # {"samples": torch.Tensor}
        timestep: float,
        guidance_scale: float,
        prompt_embeds: list | None = None,      # Conditioning — T5/CLIP text embeds
        pooled_embeds: str | None = None,       # BOUNDS placeholder (pooled CLIP)
        img_ids: torch.Tensor | None = None,    # RoPE positional image IDs [B, seq_len, 3]
    ) -> Tuple[dict]:
        """Execute single-step transformer forward on packed latents.

        Args:
            lotus_model: Lotus2ModelState with active PEFT adapter already switched.
            packed_latents: LATENT dict whose ``samples`` key holds [B, T_seq, 4C].
            timestep: Scaled timestep value (already divided by 1000).
            guidance_scale: Classifier-free diffusion guidance scale.
            prompt_embeds: Optional T5/CLIP text conditioning embeddings.
            pooled_embeds: Optional pooled CLIP projections.
            img_ids: Optional RoPE positional image IDs for latent tokens [B, seq_len, 3].

        Returns:
            Single-element tuple containing a LATENT dict with the output tensor.
        """
        # ---- a) Extract packed tensor from LATENT dict ----
        if not isinstance(packed_latents, dict) or "samples" not in packed_latents:
            raise ValueError(
                "packed_latents must be a dict containing 'samples' key "
                f"(got {type(packed_latents)} with keys {list(packed_latents.keys()) if isinstance(packed_latents, dict) else 'N/A'})"
            )

        x = packed_latents["samples"]  # [B, T_seq, 4C]

        if x.dim() != 3:
            raise ValueError(
                f"packed_latents['samples'] must be a 3D tensor [B, T_seq, C], "
                f"got shape {list(x.shape)} ({x.dim()}D)"
            )

        batch_size = x.shape[0]
        device = lotus_model.device if hasattr(lotus_model, 'device') else x.device

        # ---- b) Get transformer reference ----
        transformer = getattr(lotus_model, 'transformer', None)
        if transformer is None:
            raise AttributeError(
                "lotus_model has no 'transformer' attribute. "
                "Ensure Lotus2PeftLoader (or equivalent) populated the model state correctly."
            )

        # Force this stage to use core_predictor so stale detail_sharpener state cannot leak in.
        Lotus2AdapterSwitcher._force_active_adapter(transformer, "core_predictor")
        lotus_model.active_adapter = "core_predictor"

        weight_dtype = transformer.dtype

        # ---- c) Log active adapter and key shapes ----
        workflow_active_adapter = getattr(lotus_model, "active_adapter", None)
        peft_active_adapter = getattr(transformer, "_active_adapter", None)

        actual_active_adapter = workflow_active_adapter or peft_active_adapter
        available_adapters = list(getattr(transformer, "peft_config", {}).keys()) or []

        logger.info(
            f"Lotus2: Transformer {'reusing cache' if actual_active_adapter else 'loading'} - "
            f"mode={actual_active_adapter}, peft_mode={peft_active_adapter}, available={available_adapters}"
        )

        # ---- d) Build guidance tensor (FLUX uses config.guidance_embeds check) ----
        if getattr(transformer.config, 'guidance_embeds', False):
            guidance = torch.full([1], guidance_scale, device=device, dtype=weight_dtype)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        # ---- e) Handle prompt/pooled embeds extraction from ComfyUI CONDITIONING format ----
        if prompt_embeds is not None and isinstance(prompt_embeds, list) and len(prompt_embeds) > 0:
            enc_hidden_states = (
                prompt_embeds[0][0] if isinstance(prompt_embeds[0], (list, tuple)) else prompt_embeds[0]
            )
        else:
            text_len = getattr(transformer.config, 'max_position_embeddings', 512)
            hidden_dim = getattr(transformer.config, 'joint_attention_dim', 4096)
            enc_hidden_states = torch.zeros(
                batch_size, text_len, hidden_dim,
                device=device, dtype=weight_dtype
            )

        if pooled_embeds is not None and isinstance(pooled_embeds, list) and len(pooled_embeds) > 0:
            proj = (
                pooled_embeds[0][0] if isinstance(pooled_embeds[0], (list, tuple)) else pooled_embeds[0]
            )
        elif pooled_embeds is not None:
            try:
                proj = torch.as_tensor(pooled_embeds)
            except Exception:
                hidden_dim_p = getattr(transformer.config, 'pooled_projection_dim', 768)
                proj = torch.zeros(batch_size, hidden_dim_p, device=device, dtype=weight_dtype)
        else:
            hidden_dim_p = getattr(transformer.config, 'pooled_projection_dim', 768)
            proj = torch.zeros(batch_size, hidden_dim_p, device=device, dtype=weight_dtype)

        enc_hidden_states = enc_hidden_states.to(device=device, dtype=weight_dtype)
        proj = proj.to(device=device, dtype=weight_dtype)

        # ---- f) Create text_ids (position IDs for CLIP+T5 tokens) ----
        txt_ids = torch.zeros(
            batch_size, enc_hidden_states.shape[1], 3,
            device=device, dtype=weight_dtype
        )

        # ---- g) Handle img_ids (RoPE positional image IDs) ----
        packed_seq_len = x.shape[1]
        if img_ids is not None:
            img_ids_tensor = img_ids.to(device=device, dtype=weight_dtype)
        else:
            logger.warning(
                "No img_ids provided to Lotus2RawTransformerForward. "
                "Using zero tensor [%d, %d, 3] as fallback. "
                "Connect the IMG_IDS output from Lotus2LatentPacker (mode='pack') for correct results.",
                batch_size, packed_seq_len
            )
            img_ids_tensor = torch.zeros(
                batch_size, packed_seq_len, 3,
                device=device, dtype=weight_dtype
            )

        # ---- h) Call transformer (mirrors pipeline.py ~line 140 exactly) ----
        logger.info("Lotus2: Transformer loading active adapter")
        timestep_tensor = torch.full((batch_size,), timestep, dtype=weight_dtype, device=device)

        output_tensor = transformer(
            hidden_states=x.to(device=device, dtype=weight_dtype),
            timestep=timestep_tensor,
            guidance=guidance,
            pooled_projections=proj,
            encoder_hidden_states=enc_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids_tensor,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        # ---- i) Return as LATENT dict (standard ComfyUI format: {"samples": tensor}) ----
        result_dict = {"samples": output_tensor.clone().detach()}
        for key in ("_height", "_width"):
            if key in packed_latents:
                result_dict[key] = packed_latents[key]

        return (result_dict,)
