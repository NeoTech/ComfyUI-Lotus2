"""ComfyUI node: Flux VAE encode normalization for Lotus-2 decomposed workflows.

Standard ComfyUI VAEEncode outputs normal VAE latent samples. Diffusers' Lotus-2
pipeline then applies FLUX-specific scale/shift normalization before packing:

    rgb_latents = self.vae.encode(rgb_in).latent_dist.sample()
    rgb_latents = (rgb_latents - shift_factor) * scaling_factor

This node performs that exact transform so packed transformer latents match the
reference pipeline's latent-space distribution.
"""

import logging

import torch

from .lotus2_utils import FLUX_VAE_SCALING_FACTOR, FLUX_VAE_SHIFT_FACTOR

logger = logging.getLogger(__name__)


class Lotus2FluxVaeEncode:
    """Normalize ComfyUI VAE-encoded samples for Flux transformer packing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "image/lotus2_decomposed"

    def encode(self, samples: torch.Tensor, vae):
        """Encode image pixels and apply Flux VAE scale/shift normalization."""
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"samples must be a torch.Tensor IMAGE, got {type(samples)}")

        logger.info("Lotus2: VAE Encode loading model")
        encoded = vae.encode(samples.to(vae.device))
        latent_dist = getattr(encoded, "latent_dist", None)

        if hasattr(latent_dist, "sample"):
            latents = latent_dist.sample()
        elif isinstance(encoded, torch.Tensor):
            latents = encoded
        else:
            raise TypeError(
                "Unsupported VAE encode output. Expected tensor or object with .latent_dist.sample()."
            )

        logger.info("Lotus2: VAE Encode reusing cache")
        scaling_factor = self._get_vae_scaling_factor(vae)
        shift_factor = self._get_vae_shift_factor(vae)
        normalized = (latents - shift_factor) * scaling_factor

        return ({"samples": normalized.clone().detach()},)

    @staticmethod
    def _get_vae_scaling_factor(vae) -> float:
        """Return runtime VAE config scale factor or FLUX default fallback."""
        config = getattr(vae, "config", None)
        value = getattr(config, "scaling_factor", None)
        if value is not None:
            return float(value)

        logger.warning(
            "Loaded VAE has no config.scaling_factor; using Flux default %.4f",
            FLUX_VAE_SCALING_FACTOR,
        )
        return FLUX_VAE_SCALING_FACTOR

    @staticmethod
    def _get_vae_shift_factor(vae) -> float:
        """Return runtime VAE config shift factor or FLUX default fallback."""
        config = getattr(vae, "config", None)
        value = getattr(config, "shift_factor", None)
        if value is not None:
            return float(value)

        logger.warning(
            "Loaded VAE has no config.shift_factor; using Flux default %.4f",
            FLUX_VAE_SHIFT_FACTOR,
        )
        return FLUX_VAE_SHIFT_FACTOR


__all__ = ["Lotus2FluxVaeEncode"]
