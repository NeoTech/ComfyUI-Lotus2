"""ComfyUI node: Load Lotus-2 PEFT Adapters + Scheduler.

Loads a base FluxTransformer2DModel, creates a FlowMatchEulerDiscreteScheduler,
then attaches two named PEFT adapters (core_predictor, detail_sharpener) and an
LCM module using utilities from lotus2_utils. Returns the model state object and
scheduler so downstream nodes can use them in a decomposed workflow.

This is part of "Option A" — full visual workflow decomposition.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


class _NoOpProgressBar:
    """Fallback progress bar used when ComfyUI is not available."""

    def __init__(self, total: int):
        self.total = total
        self.value = 0

    def update(self, amount: int = 1) -> None:
        self.value += amount


def _create_model_loading_progress_bar(total: int):
    """Create a ComfyUI progress bar when running inside ComfyUI."""
    try:
        from comfy.utils import ProgressBar as ComfyProgressBar
        return ComfyProgressBar(total)
    except Exception:
        return _NoOpProgressBar(total)


def _update_model_loading_progress(progress_bar, label: str) -> None:
    """Log and advance the model-loading progress bar."""
    logger.info(label)
    try:
        progress_bar.update(1)
    except Exception as e:
        logger.debug("Failed to update ComfyUI progress bar: %s", e)


# ============================================================
# Lotus2ModelState — shared data object across the node graph
# ============================================================

@dataclass
class Lotus2ModelState:
    """Carries transformer + LCM + metadata through the workflow.

    Attributes:
        transformer: FluxTransformer2DModel with PEFT adapters attached.
        lcm_module: LocalContinuityModule (or None if not yet loaded).
        task_name: "depth" or "normal" — which adapter set is active.
        scheduler: FlowMatchEulerDiscreteScheduler for the diffusion steps.
        device: Current torch.device of the model parameters.
    """

    transformer = None      # FluxTransformer2DModel with adapters
    lcm_module = None       # LocalContinuityModule | None
    task_name: str = ""
    scheduler = None        # FlowMatchEulerDiscreteScheduler
    device: torch.device | str = ""

    _ref_count: int = 0     # Active workflow references to this cached model


# ============================================================
# Private helpers
# ============================================================

def _get_or_create_model_state(
    pretrained_model_name_or_path: str,
    task_name: str,
) -> Lotus2ModelState:
    """Load base transformer, scheduler, and PEFT+LCM adapters for a Lotus-2 task.

    Args:
        pretrained_model_name_or_path: HF model ID or local path (e.g. "black-forest-labs/FLUX.1-dev").
        task_name: One of "depth" or "normal".

    Returns:
        Fully-populated Lotus2ModelState ready for inference.
    """
    progress_bar = _create_model_loading_progress_bar(total=3)

    # --- Device & dtype --------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.warning(
            "CUDA not available — falling back to CPU. "
            "Inference will be extremely slow on FLUX-sized models."
        )
        device = torch.device("cpu")

    weight_dtype = torch.bfloat16

    # --- Scheduler -------------------------------------------------------------
    _update_model_loading_progress(progress_bar, "Loading scheduler")
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            num_train_timesteps=10,
        )
        _update_model_loading_progress(progress_bar, "Loaded scheduler; loading transformer")
    except Exception as e:
        raise RuntimeError(f"Failed to load scheduler from '{pretrained_model_name_or_path}': {e}") from e

    # --- Base transformer ------------------------------------------------------
    _update_model_loading_progress(progress_bar, "Loaded scheduler; loading transformer")
    try:
        from diffusers import FluxTransformer2DModel

        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
        )
        transformer.requires_grad_(False)
        transformer.to(device=device, dtype=weight_dtype)
        _update_model_loading_progress(progress_bar, "Loaded transformer; loading PEFT/LCM adapters")
    except Exception as e:
        raise RuntimeError(f"Failed to load transformer from '{pretrained_model_name_or_path}': {e}") from e

    # --- PEFT adapters + LCM ---------------------------------------------------
    _update_model_loading_progress(progress_bar, "Loaded transformer; loading PEFT/LCM adapters")
    try:
        from .lotus2_utils import load_lora_and_lcm_weights_for_task

        transformer, lcm_module = load_lora_and_lcm_weights_for_task(
            transformer,
            task_name=task_name,
            core_predictor_model_path=None,      # auto-download from HF
            lcm_model_path=None,                  # auto-download from HF
            detail_sharpener_model_path=None,     # auto-download from HF
        )
        _update_model_loading_progress(progress_bar, "Loaded PEFT/LCM adapters")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PEFT/LCM adapters for task='{task_name}': {e}"
        ) from e

    # --- Assemble state --------------------------------------------------------
    state = Lotus2ModelState()
    state.transformer = transformer.to(device)
    state.lcm_module = lcm_module
    state.scheduler = noise_scheduler
    state.task_name = task_name
    state.device = device
    return state


# ============================================================
# Node class
# ============================================================

class Lotus2PeftLoader:
    """Load base FLUX model + attach PEFT adapters for depth or normal estimation.

    Caches results by (model_path, task_name) to avoid re-loading on subsequent calls.
    
    IMPORTANT: The transformer's set_adapter() is stateful and in-place. Do NOT share
    the LOTUS_MODEL output between parallel workflow branches that need different
    adapters simultaneously.
    """

    # Class-level cache keyed by (pretrained_model_name_or_path, task_name)
    _CACHE = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_model_name_or_path": (
                    "STRING",
                    {"default": "black-forest-labs/FLUX.1-dev"},
                ),
                "task_name": ("depth,normal".split(","),),
            }
        }

    RETURN_TYPES = ("LOTUS_MODEL",)
    RETURN_NAMES = ("lotus_model",)
    FUNCTION = "load"
    CATEGORY = "image/lotus2_decomposed"

    def load(self, pretrained_model_name_or_path: str, task_name: str):
        """Load or return cached Lotus-2 PEFT model state.

        Increments ref_count on cache hit so cleanup knows when all users are done.

        Returns:
            Tuple containing a single Lotus2ModelState instance.
        """
        cache_key = (pretrained_model_name_or_path, task_name)

        if cache_key in self._CACHE:
            logger.info("Lotus2: Peft reusing cache")
            self._CACHE[cache_key]._ref_count += 1
            return (self._CACHE[cache_key],)

        logger.info("Lotus2: Peft Loading models")
        state = _get_or_create_model_state(pretrained_model_name_or_path, task_name)
        state._ref_count = 1
        self._CACHE[cache_key] = state
        return (state,)