"""ComfyUI-Lotus2 custom node package.

Registers Lotus-2 nodes with ComfyUI's custom-node system:
  - Single monolithic infer node (Option B fallback)
  - Decomposed PEFT loader + sub-nodes (Option A visual workflow)
"""

from .lotus2_infer_node import Lotus2Infer
from .lotus2_peft_loader import Lotus2PeftLoader, Lotus2ModelCleanup
from .lotus2_lcm_inference import Lotus2LcmInference
from .lotus2_adapter_switcher import Lotus2AdapterSwitcher
from .lotus2_latent_packer import Lotus2LatentPacker
from .lotus2_raw_transformer_forward import Lotus2RawTransformerForward
from .lotus2_packed_sampler import Lotus2PackedSampler
from .lotus2_flux_vae_decode import Lotus2FluxVaeDecode

NODE_CLASS_MAPPINGS = {
    "Lotus-2 Infer": Lotus2Infer,
    "Load-Lotus2-PEFT": Lotus2PeftLoader,
    "Lotus-2 LCM Inference": Lotus2LcmInference,
    "Lotus2AdapterSwitcher": Lotus2AdapterSwitcher,
    "Lotus-2 Latent Packer": Lotus2LatentPacker,
    "Lotus-2 Raw Transformer Forward": Lotus2RawTransformerForward,
    "Lotus-2 Packed Sampler": Lotus2PackedSampler,
    "Lotus-2 Flux VAE Decode": Lotus2FluxVaeDecode,
    "Lotus-2 Cleanup Cache": Lotus2ModelCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lotus-2 Infer": "Lotus2: Infer",
    "Load-Lotus2-PEFT": "Lotus2: Load PEFT Adapters",
    "Lotus-2 LCM Inference": "Lotus2: LCM Inference",
    "Lotus2AdapterSwitcher": "Lotus2: Switch Lotus Adapter",
    "Lotus-2 Latent Packer": "Lotus2: Pack/Unpack Flux Latents",
    "Lotus-2 Raw Transformer Forward": "Lotus2: Raw Transformer Forward",
    "Lotus-2 Packed Sampler": "Lotus2: Packed Latent Denoiser",
    "Lotus-2 Flux VAE Decode": "Lotus2: Flux VAE Decode",
    "Lotus-2 Cleanup Cache": "Lotus2: Release Model Memory"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

