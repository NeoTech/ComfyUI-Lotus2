"""ComfyUI node: LCM (Local Continuity Module) Inference — Option A decomposed workflow.

Applies the `x + lcm(x)` residual operation on unpacked spatial latents [B,C,H,W].
Sits between core_predictor unpack and detail_sharpener pack stages in the pipeline.
"""

import logging

logger = logging.getLogger(__name__)


class Lotus2LcmInference:
    """Apply LocalContinuityModule as a residual correction on unpacked latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_model": ("LOTUS_MODEL",),
                "latents": ("LATENT",),
            },
            "optional": {
                "device_override": (
                    "STRING",
                    {"default": ""},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_lcm"
    CATEGORY = "image/lotus2_decomposed"

    def apply_lcm(self, lotus_model, latents: dict, device_override: str = "") -> tuple:
        """Apply LCM residual to the unpacked spatial latents.

        Pipeline position (from Lotus-2/pipeline.py):
            core_predictor forward → _unpack_latents → **LCM here** → _pack_latents → detail_sharpener loop

        Args:
            lotus_model: Lotus2ModelState with an attached lcm_module.
            latents: ComfyUI LATENT dict; ``latents["samples"]`` is [B,C,H,W].
            device_override: Optional device string; defaults to model's device.

        Returns:
            Tuple containing a single LATENT dict with the refined samples tensor.
        """
        # --- 1. Validate lotus_model -------------------------------------------
        if not hasattr(lotus_model, "lcm_module") or lotus_model.lcm_module is None:
            raise RuntimeError(
                "LCM module is not loaded on this LOTUS_MODEL. "
                "Ensure you used Lotus2PeftLoader (which loads LCM during init)."
            )

        logger.info("Lotus2: LCM reusing cache")
        lcm_module = lotus_model.lcm_module

        # --- 2. Extract samples tensor -----------------------------------------
        if "samples" not in latents:
            raise KeyError("LATENT dict missing 'samples' key")

        x = latents["samples"]  # [B, C, H, W] unpacked spatial latents

        if x.dim() != 4:
            raise ValueError(
                f"LCM expects 4D [B,C,H,W] spatial latents, got shape {x.shape}. "
                "Did you forget to Unpack the packed transformer output first?"
            )

        # --- 3. Resolve device -------------------------------------------------
        if device_override:
            import torch as _torch
            device = _torch.device(device_override)
        else:
            device = lotus_model.device

        x = x.to(device=device)

        # Ensure LCM module parameters are on the same device (handles override path).
        current_lcm_device = lcm_module.lcm[0].weight.device
        if str(current_lcm_device) != str(device):
            lcm_module.to(device=device)

        logger.info("Lotus2: LCM loading module")
        # --- 4. LCM forward pass (handles dtype alignment internally) ----------
        refined = lcm_module(x.clone().detach())  # residual: x + lcm_conv_block(x)

        # --- 5. Return as ComfyUI LATENT dict -----------------------------------
        return ({"samples": refined.clone().detach()},)