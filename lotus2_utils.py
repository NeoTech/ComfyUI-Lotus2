"""Shared utilities for Lotus-2 ComfyUI nodes.

This module provides common classes and functions used across multiple
Lotus-2 integration points — it is NOT a node itself (no INPUT_TYPES / RETURN_TYPES).

Sections:
    1. LocalContinuityModule class
    2. HF model download helpers
    3. Latent pack/unpack wrappers (FluxPipeline static methods)
    4. VAE constants
    5. PEFT adapter loading utilities
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# Section 1: LocalContinuityModule class
# ============================================================

class LocalContinuityModule(nn.Module):
    """Local Continuity Module (LCM) — residual conv block for spatial feature refinement.
    
    Adapted from Lotus-2/infer.py Local_Continuity_Module with cleaner naming.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


# ============================================================
# Section 2: HF model download helpers
# ============================================================

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning(
        "huggingface_hub not installed — auto-download will fail. "
        "Install via: pip install huggingface_hub"
    )

# --- Default repo and filename maps -------------------------------------------
DEFAULT_REPO_NAME = "jingheya/Lotus-2"

CORE_PREDICTOR_FILENAME = {
    "depth": "lotus-2_core_predictor_depth.safetensors",
    "normal": "lotus-2_core_predictor_normal.safetensors",
}

LCM_FILENAME = {
    "depth": "lotus-2_lcm_depth.safetensors",
    "normal": "lotus-2_lcm_normal.safetensors",
}

DETAIL_SHARPENER_FILENAME = {
    "depth": "lotus-2_detail_sharpener_depth.safetensors",
    "normal": "lotus-2_detail_sharpener_normal.safetensors",
}


def get_model_path(model_path, repo_id: str, filename: str):
    """Return local path for a model — downloads from HF when *model_path* is None.

    Args:
        model_path: Existing local path or ``None`` (triggers download).
        repo_id: HuggingFace repository ID (e.g., `"jingheya/Lotus-2"`).
        filename: Filename inside the repository.

    Returns:
        Absolute path to the downloaded / existing file.

    Raises:
        ImportError: If *model_path* is None and huggingface_hub isn't installed.
        RuntimeError: If download fails for any reason.
    """
    if model_path is not None:
        return os.path.abspath(model_path)

    if not HF_AVAILABLE:
        raise ImportError(
            f"huggingface_hub is required to auto-download '{filename}'. "
            "Install via: pip install huggingface_hub"
        )

    logger.info("Downloading %s from %s", filename, repo_id)

    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)

        repo_path = _hf_snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )

        full_path = os.path.join(repo_path, filename)

        if not os.path.exists(full_path):
            # Walk in case the file landed inside a sub-directory.
            for root, _dirs, files in os.walk(repo_path):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    break
            else:
                raise FileNotFoundError(
                    f"'{filename}' not found after downloading '{repo_id}'"
                )

        logger.info("Model saved to %s", full_path)
        return full_path

    except Exception as e:
        raise RuntimeError(f"HF download failed for {repo_id}/{filename}: {e}") from e


def _load_model_weights(path: str | Path):
    """Load model weights — auto-detects format (safetensors first, torch pickle fallback)."""
    path = Path(path)

    # 1. Try safetensors
    try:
        import safetensors.torch as sf_torch
    except ImportError:
        raise ImportError(
            "safetensors is required to load Lotus-2 weights. "
            "Install via: pip install safetensors"
        ) from None

    try:
        sd = sf_torch.load_file(str(path))
        logger.info("Loaded %s via [safetensors] (%d keys)", path, len(sd))
        return sd
    except Exception as e_sf:
        logger.warning(
            "Safetensors load failed for %s (%s) — trying torch.load fallback",
            path,
            type(e_sf).__name__,
        )

    # 2. Fallback: torch pickle (original infer.py uses this for LCM weights)
    try:
        import torch as _torch
        sd = _torch.load(str(path), map_location="cpu", weights_only=True)
        logger.info("Loaded %s via [torch.load] (%d keys)", path, len(sd))
        return sd
    except Exception as e_torch:
        raise RuntimeError(
            f"Failed to load '{path}' — safetensors: {e_sf}, torch: {e_torch}"
        ) from e_sf


# ============================================================
# Section 3: Latent pack/unpack wrappers (FluxPipeline)
# ============================================================

def _import_diffusers_flux():
    """Lazy import of Diffuser's FluxPipeline to avoid startup cost."""
    try:
        from diffusers import FluxPipeline as _DiffusersFluxPipeline
    except ImportError:
        raise ImportError(
            "diffusers (with FLUX support) is required for latent pack/unpack. "
            "Install via: pip install diffusers"
        ) from None
    return _DiffusersFluxPipeline


def pack_latents(latents: torch.Tensor):
    """Pack [B, C, H, W] spatial latents → [B, T_seq, 4C] packed format.

    Splits the spatial dimensions into 2×2 patches and flattens them (FLUX's
    internal representation for DiT processing).

    Args:
        latents: Tensor of shape ``(batch_size, num_channels, height, width)``.

    Returns:
        Packed tensor of shape ``(batch_size, seq_len, 4 * num_channels)``.
    """
    FluxPipeline = _import_diffusers_flux()
    batch_size = latents.shape[0]
    height, width = latents.shape[2], latents.shape[3]

    packed_latents = FluxPipeline._pack_latents(
        latents,
        batch_size=batch_size,
        num_channels_latents=latents.shape[1],
        height=height,
        width=width,
    )
    return packed_latents


def unpack_latents(packed_latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    """Unpack [B, T_seq, 4C] → [B, C_out, H_out, W_out].

    Args:
        packed_latents: Packed tensor from ``pack_latents()``.
        height: Original latent height (before packing).
        width: Original latent width (before packing).
        vae_scale_factor: VAE downscale factor (FLUX = 8).

    Returns:
        Unpacked spatial latents ready for VAE decode.
    """
    FluxPipeline = _import_diffusers_flux()
    unpacked = FluxPipeline._unpack_latents(
        packed_latents,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
    )
    return unpacked


def prepare_latent_image_ids(batch_size: int, latent_height: int, latent_width: int, device, dtype) -> torch.Tensor:
    """Prepare RoPE positional image IDs for FLUX.

    Args:
        batch_size: Number of samples.
        latent_height: Height in packed coordinates (``latent_H // 2``).
        latent_width: Width in packed coordinates (``latent_W // 2``).
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor ``[batch_size, H*W, 3]`` with (x, y, t) positional IDs for RoPE.
    """
    FluxPipeline = _import_diffusers_flux()
    image_ids = FluxPipeline._prepare_latent_image_ids(
        batch_size, latent_height, latent_width, device, dtype
    )
    return image_ids


# ============================================================
# Section 4: VAE constants (FLUX)
# ============================================================

# Shift and scale factors from FLUX's vae.config — used when encoding / decoding.
FLUX_VAE_SHIFT_FACTOR = 0.0609
FLUX_VAE_SCALING_FACTOR = 0.3611


# ============================================================
# Section 5: PEFT adapter loading
# ============================================================

_TARGET_LORA_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


def load_lora_and_lcm_weights_for_task(
    transformer,
    task_name: str = "depth",
    core_predictor_model_path=None,
    lcm_model_path=None,
    detail_sharpener_model_path=None,
):
    """Load PEFT adapters and LCM for a given Lotus-2 task.

    Attaches two named LoRA adapters to the transformer:
        - ``"core_predictor"``  (stage‑1 geometry predictor)
        - ``"detail_sharpener"``  (stage‑2 refinement)

    Also instantiates and loads weights into a :class:`LocalContinuityModule`.

    Args:
        transformer: A FLUX Transformer2DModel instance.
        task_name: Either `"depth"` or `"normal"`. Controls LoRA rank (128 / 256).
        core_predictor_model_path: Local path to ``core_predictor`` safetensors, or None to auto-download.
        lcm_model_path: Local path to LCM weights (.safetensors), or None to auto-download.
        detail_sharpener_model_path: Local path to sharpener safetensors, or None to auto-download.

    Returns:
        Tuple ``(transformer, local_continuity_module)`` — both ready for inference.
    """
    try:
        from peft import LoraConfig, set_peft_model_state_dict
    except ImportError:
        raise ImportError(
            "peft is required to load LoRA adapters. Install via: pip install peft"
        ) from None

    try:
        from diffusers.utils import convert_unet_state_dict_to_peft
    except ImportError:
        raise ImportError(
            "diffusers >= 0.32 (with PEFT utilities) is required. "
            "Install via: pip install --upgrade diffusers"
        ) from None

    if task_name not in ("depth", "normal"):
        raise ValueError(f"task_name must be 'depth' or 'normal', got '{task_name}'")

    lora_rank = 128 if task_name == "depth" else 256
    device = transformer.device
    weight_dtype = transformer.dtype

    # ---- Resolve (auto-download) all three model paths -----------------------
    core_predictor_model_path = get_model_path(
        core_predictor_model_path, DEFAULT_REPO_NAME, CORE_PREDICTOR_FILENAME[task_name]
    )
    lcm_model_path = get_model_path(
        lcm_model_path, DEFAULT_REPO_NAME, LCM_FILENAME[task_name]
    )
    detail_sharpener_model_path = get_model_path(
        detail_sharpener_model_path, DEFAULT_REPO_NAME, DETAIL_SHARPENER_FILENAME[task_name]
    )

    # ---- 1. Core predictor LoRA ---------------------------------------------
    core_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=_TARGET_LORA_MODULES,
    )
    transformer.add_adapter(core_lora_config, adapter_name="core_predictor")

    core_state_dict = _load_model_weights(core_predictor_model_path)
    # Strip the "transformer." prefix that diffusers pipelines wrap around keys.
    core_transformer_sd = {
        k.replace("transformer.", ""): v
        for k, v in core_state_dict.items()
        if k.startswith("transformer.")
    }
    core_transformer_sd = convert_unet_state_dict_to_peft(core_transformer_sd)

    incompatible_keys = set_peft_model_state_dict(
        transformer, core_transformer_sd, adapter_name="core_predictor"
    )
    _warn_incompatible_keys(incompatible_keys, "core_predictor")

    for name, param in transformer.named_parameters():
        if "core_predictor" in name:
            param.requires_grad = False

    logger.info("Loaded LoRA weights for [core predictor] (rank=%d).", lora_rank)

    # ---- 2. Local Continuity Module (LCM) -----------------------------------
    num_lcm_channels = transformer.config.in_channels // 4
    local_continuity_module = LocalContinuityModule(num_lcm_channels)
    lcm_state_dict = _load_model_weights(lcm_model_path)
    local_continuity_module.load_state_dict(lcm_state_dict)
    local_continuity_module.requires_grad_(False)
    local_continuity_module.to(device=device, dtype=weight_dtype)
    logger.info("Loaded weights for [local continuity module (LCM)].")

    # ---- 3. Detail sharpener LoRA -------------------------------------------
    sharpener_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=_TARGET_LORA_MODULES,
    )
    transformer.add_adapter(sharpener_lora_config, adapter_name="detail_sharpener")

    sharpener_state_dict = _load_model_weights(detail_sharpener_model_path)
    sharpener_transformer_sd = {
        k.replace("transformer.", ""): v
        for k, v in sharpener_state_dict.items()
        if k.startswith("transformer.")
    }
    sharpener_transformer_sd = convert_unet_state_dict_to_peft(sharpener_transformer_sd)

    incompatible_keys = set_peft_model_state_dict(
        transformer, sharpener_transformer_sd, adapter_name="detail_sharpener"
    )
    _warn_incompatible_keys(incompatible_keys, "detail_sharpener")

    for name, param in transformer.named_parameters():
        if "detail_sharpener" in name:
            param.requires_grad = False

    logger.info("Loaded LoRA weights for [detail sharpener] (rank=%d).", lora_rank)

    return transformer, local_continuity_module


def _warn_incompatible_keys(incompatible_keys, adapter_name: str):
    """Log a warning when PEFT state-dict loading has unexpected keys."""
    if incompatible_keys is not None:
        unexpected = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected:
            logger.warning(
                "Loading adapter '%s' had unexpected keys: %s",
                adapter_name,
                unexpected,
            )
