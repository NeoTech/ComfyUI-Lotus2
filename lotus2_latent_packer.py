"""ComfyUI node: Pack/Unpack Flux Latents for Lotus-2 decomposed workflow.

Wraps `FluxPipeline._pack_latents()` and its inverse to convert between spatial 
latents `[B, C, H, W]` and packed transformer input `[B, T_seq, 64]`. Also produces
the `_prepare_latent_image_ids` tensor needed by the transformer forward pass.

This is Step 5 of the todo.md plan — enables decomposed core_predictor / 
detail_sharpener stages where each stage receives a different latent format.
"""

import logging

import torch

from .lotus2_utils import pack_latents, unpack_latents, prepare_latent_image_ids

logger = logging.getLogger(__name__)

VAE_SCALE_FACTOR = 8  # FLUX uses factor-8 VAE


class Lotus2LatentPacker:
    """Pack spatial latents into 1D sequence or unpack them back to [B,C,H,W].

    Pipeline origin (from Lotus-2/pipeline.py):
        - _pack_latents(): Converts noise/ref_image from [B,4,H,W] -> [B,T_seq,64], 
          also produces image_ids via `_prepare_latent_image_ids`.
        - _unpack_latents(): Inverse — reshapes transformer output back to spatial.

    Pack mode stores height/width metadata in the returned LATENT dict so unpack 
    can auto-detect dimensions without manual input.
    """

    MODES = ("pack", "unpack")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "mode": (cls.MODES,),
            },
            "optional": {
                "height": ("INT", {"default": 0}),
                "width": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMG_IDS")
    FUNCTION = "execute"
    CATEGORY = "image/lotus2_decomposed"

    def execute(
        self,
        latents: dict,
        mode: str,
        height: int = 0,
        width: int = 0,
    ) -> tuple:
        """Execute pack or unpack operation on the provided latents.

        Args:
            latents: ComfyUI LATENT dict; ``latents["samples"]`` is [B,C,H,W] for 
                     pack mode, or contains packed metadata for unpack mode.
            mode: "pack" to convert spatial->sequence, "unpack" for sequence->spatial.
            height: Optional override for latent height (used in unpack when not 
                    stored in packed metadata). Default 0 means auto-detect.
            width: Optional override for latent width.

        Returns:
            Tuple of (LATENT dict, IMG_IDS tensor or empty dict). In pack mode the 
            first element contains packed samples with stored height/width; in unpack 
            mode it contains spatial [B,C,H,W] latents and IMG_IDS is an empty dict.
        """
        x = latents["samples"]

        if mode == "pack":
            return self._pack(x)
        elif mode == "unpack":
            return self._unpack(latents, height, width)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {self.MODES}")

    def _pack(self, x: torch.Tensor) -> tuple:
        """Pack spatial [B,C,H,W] latents into sequence format and produce image IDs."""
        if x.dim() != 4:
            raise ValueError(
                f"Pack mode expects 4D tensor [B,C,H,W], got {x.dim()}D with shape {tuple(x.shape)}. "
                "Ensure the input is a spatial LATENT dict, not already-packed data."
            )

        B, C, H, W = x.shape
        logger.info("Packing latents: %s -> packed sequence", tuple(x.shape))

        packed_tensor = pack_latents(x)
        logger.info(
            "Packed result shape: %s (device=%s)", tuple(packed_tensor.shape), packed_tensor.device
        )

        # Store H/W metadata for unpack mode
        result_dict = {
            "samples": packed_tensor, 
            "_height": H * VAE_SCALE_FACTOR,
            "_width": W * VAE_SCALE_FACTOR,
        }

        # Image IDs use half-size coords because FLUX patches 2x2 blocks
        img_ids = prepare_latent_image_ids(
            batch_size=B,
            latent_height=H // 2,
            latent_width=W // 2,
            device=x.device,
            dtype=x.dtype,
        )
        logger.info("Image IDs shape: %s", tuple(img_ids.shape))

        return (result_dict, img_ids)

    def _unpack(self, latents: dict, height_param: int, width_param: int) -> tuple:
        """Unpack packed sequence latents back to spatial [B,C,H,W]."""
        x = latents["samples"]

        # Resolve H/W from stored metadata or manual params
        if "_height" in latents and "_width" in latents:
            H, W = int(latents["_height"]), int(latents["_width"])
        elif height_param > 0 and width_param > 0:
            H, W = height_param, width_param
        else:
            raise RuntimeError(
                "Unpack mode needs latent dimensions. Either provide 'height' and 'width' parameters, "
                "or use the output of pack mode (which stores _height/_width metadata). "
                f"Got height={height_param}, width={width_param}, stored keys: {[k for k in latents if k.startswith('_')]}"
            )

        logger.info("Unpacking packed latents %s with H=%d, W=%d", tuple(x.shape), H, W)

        unpacked = unpack_latents(x, height=H, width=W, vae_scale_factor=VAE_SCALE_FACTOR)
        logger.info(
            "Unpacked result shape: %s (device=%s)", tuple(unpacked.shape), unpacked.device
        )

        return ({"samples": unpacked}, {})
