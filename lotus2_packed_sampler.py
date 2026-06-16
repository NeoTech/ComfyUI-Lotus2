"""ComfyUI node: Packed latent denoiser with custom sigma scheduling.

Step 7 of todo.md — implements a full denoising loop over packed latents using
user-defined sigmas (num_steps) and guidance scale. Accepts the output from
Lotus2LcmInference, applies multiple transformer forward passes via custom
sigma interpolation, and returns fully-denoised packed latents ready for
unpacking + VAE decoding.

Pipeline origin (from Lotus-2/pipeline.py):
    - The detail_sharpener denoising loop (~lines 150–180) iterates over a set of
      timesteps, calling self.transformer(...) each step then updating the latent
      state via Euler/FlowMatch scheduler steps.

This node encapsulates that entire loop so users can tune num_steps and guidance
independently from individual transformer calls (Step 6).
"""

import logging
from typing import Tuple, Optional

import torch
import numpy as np
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

# Type-hint only — Lotus2ModelState lives in lotus2_peft_loader.
try:
    from .lotus2_peft_loader import Lotus2ModelState
except ImportError:
    # Allow instantiation without full ComfyUI env (e.g. unit tests).
    Lotus2ModelState = None  # type: ignore


logger = logging.getLogger(__name__)


class Lotus2PackedSampler:
    """Multi-step denoiser for packed latents with custom sigma scheduling.

    Drives a complete Euler/FlowMatch denoising loop over the detail_sharpener
    adapter's packed latent space. The caller must have already switched to the
    ``detail_sharpener`` adapter via Lotus2AdapterSwitcher before this node runs.

    Input shape expectation for *packed_latents["samples"]*:
        [B, T_seq, 4C] where C = latent_channels (typically 64).

    Returns:
        LATENT dict containing the denoised packed tensor (same shape as input).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_model": ("LOTUS_MODEL",),
                "packed_latents": ("LATENT",),  # {"samples": torch.Tensor [B, T_seq, C]}
                "num_steps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.5}),
            },
            "optional": {
                "prompt_embeds": ("CONDITIONING",),   # ComfyUI text conditioning (T5/CLIP embeds)
                "pooled_embeds": ("STRING", {"default": ""}),  # pooled CLIP projections placeholder
                "img_ids": ("IMG_IDS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("denoised_packed_latents",)
    FUNCTION = "denoise"
    CATEGORY = "image/lotus2_decomposed"

    def denoise(
        self,
        lotus_model: "Lotus2ModelState",
        packed_latents: dict,                   # {"samples": torch.Tensor [B, T_seq, C]}
        num_steps: int,
        guidance_scale: float,
        prompt_embeds: list | None = None,      # Conditioning — T5/CLIP text embeds
        pooled_embeds: str | None = None,       # BOUNDS placeholder (pooled CLIP)
        img_ids: torch.Tensor | None = None,    # RoPE positional image IDs [B, seq_len, 3]
    ) -> Tuple[dict]:
        """Run multi-step denoising on packed latents using custom sigma scheduling.

        Args:
            lotus_model: Lotus2ModelState with detail_sharpener adapter already active.
            packed_latents: LATENT dict whose ``samples`` key holds [B, T_seq, C].
            num_steps: Number of Euler/FlowMatch denoising steps to perform.
            guidance_scale: Classifier-free diffusion guidance scale for all steps.
            prompt_embeds: Optional T5/CLIP text conditioning embeddings.
            pooled_embeds: Optional pooled CLIP projections.
            img_ids: Optional RoPE positional image IDs [B, seq_len, 3].

        Returns:
            Single-element tuple containing a LATENT dict with denoised packed tensor.
        """
        # ---- A: Extract tensor & setup ----
        if not isinstance(packed_latents, dict) or "samples" not in packed_latents:
            raise ValueError(
                "packed_latents must be a dict containing 'samples' key "
                f"(got {type(packed_latents)} with keys {list(packed_latents.keys()) if isinstance(packed_latents, dict) else 'N/A'})"
            )

        x = packed_latents["samples"]  # [B, T_seq, C]

        if x.dim() != 3:
            raise ValueError(
                f"packed_latents['samples'] must be a 3D tensor [B, T_seq, C], "
                f"got shape {list(x.shape)} ({x.dim()}D)"
            )

        batch_size = x.shape[0]
        device = lotus_model.device if hasattr(lotus_model, 'device') else x.device

        transformer = getattr(lotus_model, 'transformer', None)
        if transformer is None:
            raise AttributeError(
                "lotus_model has no 'transformer' attribute. "
                "Ensure Lotus2PeftLoader (or equivalent) populated the model state correctly."
            )

        scheduler = getattr(lotus_model, 'scheduler', None)
        if scheduler is None:
            raise AttributeError(
                "lotus_model has no 'scheduler' attribute. "
                "Ensure Lotus2PeftLoader (or equivalent) populated the model state correctly."
            )

        logger.info(
            "Lotus2PackedSampler — batch=%d, packed_seq_len=%d, num_steps=%d, guidance=%.1f",
            batch_size, x.shape[1], num_steps, guidance_scale,
        )

        # ---- B: Build guidance tensor ----
        if getattr(transformer.config, 'guidance_embeds', False):
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        # ---- C: Extract prompt/pooled embeds ----
        if prompt_embeds is not None and isinstance(prompt_embeds, list) and len(prompt_embeds) > 0:
            enc_hidden_states = (
                prompt_embeds[0][0] if isinstance(prompt_embeds[0], (list, tuple)) else prompt_embeds[0]
            )
        else:
            text_len = getattr(transformer.config, 'max_position_embeddings', 512)
            hidden_dim = getattr(transformer.config, 'joint_attention_dim', 4096)
            enc_hidden_states = torch.zeros(
                batch_size, text_len, hidden_dim,
                device=device, dtype=x.dtype
            )
            logger.info("No prompt_embeds provided — using zero tensor [%d, %d, %d]",
                        batch_size, text_len, hidden_dim)

        if pooled_embeds is not None and isinstance(pooled_embeds, list) and len(pooled_embeds) > 0:
            proj = (
                pooled_embeds[0][0] if isinstance(pooled_embeds[0], (list, tuple)) else pooled_embeds[0]
            )
        elif pooled_embeds is not None:
            try:
                proj = torch.as_tensor(pooled_embeds)
            except Exception:
                hidden_dim_p = getattr(transformer.config, 'pooled_projection_dim', 768)
                proj = torch.zeros(batch_size, hidden_dim_p, device=device, dtype=x.dtype)
        else:
            hidden_dim_p = getattr(transformer.config, 'pooled_projection_dim', 768)
            proj = torch.zeros(batch_size, hidden_dim_p, device=device, dtype=x.dtype)
            logger.info("No pooled_embeds provided — using zero tensor [%d, %d]",
                        batch_size, hidden_dim_p)

        enc_hidden_states = enc_hidden_states.to(device=device, dtype=x.dtype)
        proj = proj.to(device=device, dtype=x.dtype)

        # ---- D: Create text_ids ----
        txt_ids = torch.zeros(
            batch_size, enc_hidden_states.shape[1], 3,
            device=device
        )

        # ---- E: Handle img_ids ----
        packed_seq_len = x.shape[1]
        if img_ids is not None:
            img_ids_tensor = img_ids.to(device=device, dtype=x.dtype)
        else:
            logger.warning(
                "No img_ids provided to Lotus2PackedSampler. "
                "Using zero tensor [%d, %d, 3] as fallback. "
                "Connect the IMG_IDS output from Lotus2LatentPacker (mode='pack') for correct results.",
                batch_size, packed_seq_len
            )
            img_ids_tensor = torch.zeros(
                batch_size, packed_seq_len, 3,
                device=device
            )

        # ---- F: Custom sigma scheduling (mirrors pipeline.py ~lines 95-108) ----
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = x.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        timesteps, _num_inference_steps = retrieve_timesteps(
            scheduler,
            num_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # ---- G: Denoising loop (mirrors pipeline.py ~lines 152-178) ----
        latents = x
        num_warmup_steps = max(len(timesteps) - len(timesteps) * scheduler.order, 0) if hasattr(scheduler, 'order') else 0

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = transformer(
                    hidden_states=latents.to(device),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=proj,
                    encoder_hidden_states=enc_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids_tensor,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    try:
                        if torch.backends.mps.is_available():
                            latents = latents.to(latents_dtype)
                    except Exception:
                        pass

                logger.info("Step %d/%d, t=%.4f", i + 1, len(timesteps), float(t))

        # ---- H: Return denoised packed latents as LATENT dict ----
        logger.info(
            "Denoising complete — output shape %s",
            list(latents.shape)
        )

        result_dict = {"samples": latents}
        for key in ("_height", "_width"):
            if key in packed_latents:
                result_dict[key] = packed_latents[key]

        return (result_dict,)
