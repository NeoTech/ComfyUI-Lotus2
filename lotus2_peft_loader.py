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
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        logger.info(
            "Loading scheduler from %s",
            pretrained_model_name_or_path,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            num_train_timesteps=10,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load scheduler from '{pretrained_model_name_or_path}': {e}") from e

    # --- Base transformer ------------------------------------------------------
    try:
        from diffusers import FluxTransformer2DModel
        logger.info(
            "Loading base transformer from %s",
            pretrained_model_name_or_path,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
        )
        transformer.requires_grad_(False)
        transformer.to(device=device, dtype=weight_dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to load transformer from '{pretrained_model_name_or_path}': {e}") from e

    # --- PEFT adapters + LCM ---------------------------------------------------
    try:
        from .lotus2_utils import load_lora_and_lcm_weights_for_task

        logger.info("Loading PEFT adapters and LCM for task='%s'", task_name)
        transformer, lcm_module = load_lora_and_lcm_weights_for_task(
            transformer,
            task_name=task_name,
            core_predictor_model_path=None,      # auto-download from HF
            lcm_model_path=None,                  # auto-download from HF
            detail_sharpener_model_path=None,     # auto-download from HF
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PEFT/LCM adapters for task='{task_name}': {e}"
        ) from e

    # --- Assemble state --------------------------------------------------------
    logger.info("Model state assembled — transformer on %s", device)
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
            logger.info("Reusing cached model state [%s].", cache_key)
            self._CACHE[cache_key]._ref_count += 1
            return (self._CACHE[cache_key],)

        state = _get_or_create_model_state(pretrained_model_name_or_path, task_name)
        state._ref_count = 1
        self._CACHE[cache_key] = state
        logger.info("New model state cached [%s].", cache_key)
        return (state,)


def _cleanup_model_state(
    pretrained_model_name_or_path: str | None = None,
    task_name: str | None = None,
):
    """Decrement ref count for a cached model; unload if zero.

    Each call decrements the matched entry's _ref_count by 1. When it reaches 
    0 the transformer + LCM are moved to CPU and removed from cache.

    Args:
        pretrained_model_name_or_path: If None, sweep ALL entries (decrement every ref).
                                        Otherwise match specific path+task combo.
        task_name: Task name ("depth"/"normal"). Ignored when pretrained is None.
    """
    keys_to_remove = []

    for cache_key, state in Lotus2PeftLoader._CACHE.items():
        # If filtering by key, skip non-matching entries
        if pretrained_model_name_or_path is not None and task_name is not None:
            if cache_key != (pretrained_model_name_or_path, task_name):
                continue

        # Decrement ref count — each cleanup call reduces by 1
        state._ref_count -= 1
        logger.info("Ref count for [%s] decremented to %d", cache_key, state._ref_count)

        if state._ref_count <= 0:
            keys_to_remove.append(cache_key)
            logger.info("Unloading model state [%s] — ref count reached 0.", cache_key)

            # Move to CPU first, then delete references
            if state.transformer is not None:
                try:
                    state.transformer.to(device="cpu")
                except Exception as e:
                    logger.warning("Error moving transformer to CPU during cleanup: %s", e)
                delattr(state, "transformer")

            if hasattr(state, "lcm_module") and state.lcm_module is not None:
                try:
                    state.lcm_module.to(device="cpu")
                except Exception as e:
                    logger.warning("Error moving LCM to CPU during cleanup: %s", e)
                delattr(state, "lcm_module")

    # Remove all zero-ref entries from cache AFTER processing (avoid dict mutation while iterating)
    for key in keys_to_remove:
        Lotus2PeftLoader._CACHE.pop(key, None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Cleanup node — explicit VRAM release
# ============================================================

class Lotus2ModelCleanup:
    """Explicitly decrement ref count and unload cached models from VRAM.

    Place this at the end of a workflow branch or before loading new tasks to 
    free up GPU memory when no longer needed.
    
    When pretrained_model_name_or_path is 'ALL', releases all 0-ref entries across every task."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_model_name_or_path": (
                    ["ALL", "black-forest-labs/FLUX.1-dev"],
                    {"default": "black-forest-labs/FLUX.1-dev"},
                ),
                "task_name": ("depth,normal".split(","),),
            },
            "optional": {
                "trigger": (
                    "*",
                    {"tooltip": "Wire from LOTUS_MODEL output to run cleanup AFTER inference."},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "cleanup"
    CATEGORY = "image/lotus2_decomposed"
    OUTPUT_NODE = True  # Signal this is a terminal cleanup node

    def cleanup(self, pretrained_model_name_or_path: str, task_name: str, trigger=None):
        """Decrement ref count and unload if all users are done."""
        path = None if pretrained_model_name_or_path == "ALL" else pretrained_model_name_or_path
        tsk = task_name if path is not None else None

        _cleanup_model_state(pretrained_model_name_or_path=path, task_name=tsk)

        return ()