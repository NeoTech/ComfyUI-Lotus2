"""ComfyUI node: Flux-specific VAE decode for Lotus-2 unpacked latents.

The decomposed sampler returns FLUX transformer-space spatial latents in the same
shape as packed/unpacked latent tensors, ``[B, C, H, W]``. Diffusers' Flux pipeline
converts those final latents into VAE decode space before calling ``vae.decode()``:

    decoded_latents = (latents / scaling_factor) + shift_factor

ComfyUI's built-in `VAEDecode` does not apply this conversion, so unpacked FLUX
latents passed directly to it can produce residual latent-domain artifacts. This
node performs the Flux VAE decode-space conversion before delegating to the loaded
ComfyUI VAE's normal decode path.
"""

import logging

from .lotus2_utils import FLUX_VAE_SCALING_FACTOR, FLUX_VAE_SHIFT_FACTOR

logger = logging.getLogger(__name__)


class Lotus2FluxVaeDecode:
    """Decode unpacked Flux latents with the required VAE scale/shift conversion.

    Args:
        samples: ComfyUI LATENT dict containing ``samples`` as spatial
            ``[B, C, H, W]`` FLUX latent tensors.
        vae: Loaded ComfyUI VAE object. The node calls ``vae.decode(...)`` directly
            so ComfyUI's normal device/dtype/tiled decode handling is preserved.

    Returns:
        IMAGE tensor ready for preview/save nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "image/lotus2_decomposed"

    def decode(self, samples: dict, vae):
        """Decode Flux latents after converting them to VAE-space."""
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError(
                "samples must be a LATENT dict containing 'samples' key "
                f"(got {type(samples)} with keys {list(samples.keys()) if isinstance(samples, dict) else 'N/A'})"
            )

        latents = samples["samples"]
        scaling_factor = self._get_vae_scaling_factor(vae)
        shift_factor = self._get_vae_shift_factor(vae)

        decoded_latents = (latents.detach() / scaling_factor) + shift_factor
        images = vae.decode(decoded_latents.cpu())

        logger.info(
            "Lotus2FluxVaeDecode — samples=%s, scaling=%.4f, shift=%.4f",
            tuple(latents.shape),
            scaling_factor,
            shift_factor,
        )

        return (images,)

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


__all__ = ["Lotus2FluxVaeDecode"]
