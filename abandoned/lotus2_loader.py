"""
Lotus-2 LoRA loaders for ComfyUI.

Uses ComfyUI's native ModelPatcher.add_patches() mechanism (NOT PEFT) to
attach the Lotus-2 LoRAs. The safetensors files use the standard diffusers
LoRA key format (lora_A / lora_B), which ComfyUI's LoRAAdapter.load()
recognizes via its `diffusers2_lora` branch.
"""

import logging
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import comfy.lora
import comfy.model_patcher
from torch import nn

logger = logging.getLogger(__name__)


REPO_ID = "jingheya/Lotus-2"

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


class LocalContinuityModule(nn.Module):
    """Local Continuity Module for smoothing patch/grid artifacts."""

    def __init__(self, num_channels):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


def _translate_lotus_keys_to_comfy(lora_state_dict, model_state_dict_keys):
    """
    Advanced Diffusers/Lotus-2 to ComfyUI key translator.
    Maps separate Q/K/V pathways to ComfyUI's unified .qkv structural blocks.
    """
    to_load = {}
    
    for k in lora_state_dict.keys():
        base_path = None
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suffix):
                base_path = k[:-len(suffix)]
                break
        
        if not base_path:
            continue

        comfy_key = base_path
        if comfy_key.startswith("transformer."):
            comfy_key = comfy_key.replace("transformer.", "diffusion_model.", 1)
        
        # 1. Map block layout structures
        if "single_transformer_blocks." in comfy_key:
            comfy_key = comfy_key.replace("single_transformer_blocks.", "single_blocks.")
        elif "transformer_blocks." in comfy_key:
            comfy_key = comfy_key.replace("transformer_blocks.", "double_blocks.")

        # 2. Map structural layers for Double Blocks
        if "double_blocks." in comfy_key:
            # --- Image Attention Stream (Mapped to ComfyUI's combined qkv) ---
            if any(x in comfy_key for x in ["attn.to_q", "attn.to_k", "attn.to_v"]):
                # ComfyUI packs image stream weights into img_attn.qkv
                comfy_key = comfy_key.split("attn.to_")[0] + "img_attn.qkv"
            elif "attn.to_out.0" in comfy_key:
                comfy_key = comfy_key.replace("attn.to_out.0", "img_attn.proj")
                
            # --- Text Attention Stream (Mapped to ComfyUI's combined qkv) ---
            elif any(x in comfy_key for x in ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]):
                # ComfyUI packs text stream weights into txt_attn.qkv
                comfy_key = comfy_key.split("attn.add_")[0].split("attn.to_")[0] + "txt_attn.qkv"
            elif "attn.to_add_out" in comfy_key:
                comfy_key = comfy_key.replace("attn.to_add_out", "txt_attn.proj")
                
            # --- Feed-Forward Neural Networks (MLPs) ---
            elif "ff.net.0.proj" in comfy_key:
                comfy_key = comfy_key.replace("ff.net.0.proj", "img_mlp.0")
            elif "ff.net.2" in comfy_key:
                comfy_key = comfy_key.replace("ff.net.2", "img_mlp.2")
            elif "ff_context.net.0.proj" in comfy_key:
                comfy_key = comfy_key.replace("ff_context.net.0.proj", "txt_mlp.0")
            elif "ff_context.net.2" in comfy_key:
                comfy_key = comfy_key.replace("ff_context.net.2", "txt_mlp.2")

        # 3. Map structural layers for Single Blocks
        elif "single_blocks." in comfy_key:
            if any(x in comfy_key for x in ["attn.to_q", "attn.to_k", "attn.to_v"]):
                comfy_key = comfy_key.split("attn.to_")[0] + "linear1"
            elif "attn.to_out.0" in comfy_key:
                comfy_key = comfy_key.replace("attn.to_out.0", "linear2")

        # 4. Verify translated path structure against internal Comfy state map
        for m_key in model_state_dict_keys:
            if m_key.startswith(comfy_key):
                to_load[base_path] = comfy_key
                break

    return to_load


def _load_lora_via_model_patcher(model_patcher, lora_path, strength=1.0):
    """
    Load a diffusers-style LoRA safetensors file and attach it to a ComfyUI
    ModelPatcher using its native add_patches mechanism.
    """
    # 1. Load the raw safetensors state dict.
    lora_sd = load_file(lora_path, device="cpu")

    # 2. Read the model's actual state dict keys for the mapping.
    model_sd_keys = set(model_patcher.model.state_dict().keys())

    # 3. Build the key mapping (LoRA prefix -> model key).
    to_load = _translate_lotus_keys_to_comfy(lora_sd, model_sd_keys)
    if not to_load:
        raise RuntimeError(
            f"[Lotus-2] No matching LoRA keys found in model. "
            f"Model has keys like: {list(model_sd_keys)[:3]}"
        )

    # 4. Use ComfyUI's load_lora to build the patch dict. This handles the
    #    diffusers2_lora format (lora_A / lora_B) natively.
    patch_dict = comfy.lora.load_lora(lora_sd, to_load)

    if not patch_dict:
        raise RuntimeError(
            f"[Lotus-2] comfy.lora.load_lora returned empty patch_dict. "
            f"to_load sample: {list(to_load.items())[:2]}"
        )

    # 5. Attach patches to the model patcher. Strength = how strongly the
    #    LoRA modifies the base weights (1.0 = full effect).
    model_patcher.add_patches(patch_dict, strength_patch=strength, strength_model=strength)
    logger.info(
        f"[Lotus-2] Loaded {len(patch_dict)} LoRA patches from {os.path.basename(lora_path)} "
        f"(strength={strength})"
    )
    return model_patcher


class LoadLotus2Adapters:
    """Loads specialized Lotus-2 LoRAs onto an incoming FLUX model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),  # From standard Checkpoint Loader
                "model_task": (["depth", "normal"], {"default": "depth"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_adapters"
    CATEGORY = "Lotus2/Loaders"

    def load_adapters(self, model, model_task, strength):
        # `model` is a comfy.model_patcher.ModelPatcher.
        # `model.model` is the underlying nn.Module (e.g. comfy.ldm.flux.model.Flux).

        core_path = hf_hub_download(
            repo_id=REPO_ID, filename=CORE_PREDICTOR_FILENAME[model_task]
        )
        sharpener_path = hf_hub_download(
            repo_id=REPO_ID, filename=DETAIL_SHARPENER_FILENAME[model_task]
        )

        # Both LoRAs are added on top of the base model. We can only have one
        # active strength at a time in ComfyUI's patching system, so the
        # two LoRAs compose multiplicatively via separate add_patches calls.
        # Since the original Lotus-2 alternates between them per stage, we
        # load both with strength=1.0 and rely on inference to either pick
        # one or the other. For a single-pass inference, loading both
        # stacked gives an approximation of the core_predictor effect.
        # For stage-aware switching, see the inference node.
        _load_lora_via_model_patcher(model, core_path, strength=strength)
        _load_lora_via_model_patcher(model, sharpener_path, strength=strength)

        logger.info(f"[Lotus-2] Injected core and sharpener adapters for task: {model_task}")
        return (model,)


class LoadLotus2LCM:
    """Downloads and constructs the Local Continuity Module weights."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_task": (["depth", "normal"], {"default": "depth"}),
                "num_channels": (
                    "INT",
                    {"default": 16, "min": 1, "max": 64,
                     "tooltip": "FLUX internal latent channel count (typically 16)"},
                ),
            }
        }

    RETURN_TYPES = ("LOTUS_LCM",)
    RETURN_NAMES = ("lcm",)
    FUNCTION = "load_lcm"
    CATEGORY = "Lotus2/Loaders"

    def load_lcm(self, model_task, num_channels=16):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        lcm_path = hf_hub_download(repo_id=REPO_ID, filename=LCM_FILENAME[model_task])

        lcm = LocalContinuityModule(num_channels)
        lcm_state_dict = torch.load(lcm_path, map_location="cpu", weights_only=True)
        lcm.load_state_dict(lcm_state_dict)

        lcm.requires_grad_(False)
        lcm.to(device=device)

        logger.info(f"[Lotus-2] Modular LCM initialized successfully for {model_task}")
        return (lcm,)