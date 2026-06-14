"""ComfyUI custom node wrapping the Lotus-2 inference pipeline."""

import os
import sys
import uuid
import logging
import tempfile

import torch
from PIL import Image
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - LotusInferNode - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("LotusInferNode")


_PIPELINES: dict = {}


def _ensure_lotus_path() -> None:
    """Prepend Lotus-2 to sys.path AND preload utils submodules into sys.modules."""
    import importlib.util

    lotus_dir = os.path.join(os.path.dirname(__file__), "Lotus-2")
    if lotus_dir in sys.path and sys.path[0] != lotus_dir:
        sys.path.remove(lotus_dir)
    sys.path.insert(0, lotus_dir)

    # Preload utils submodules into sys.modules to bypass namespace package shadowing.
    for mod_name, rel_path in [
        ("utils", os.path.join("Lotus-2", "utils", "__init__.py")),
        ("utils.image_utils", os.path.join("Lotus-2", "utils", "image_utils.py")),
        ("utils.seed_all", os.path.join("Lotus-2", "utils", "seed_all.py")),
    ]:
        if mod_name not in sys.modules:
            full_path = os.path.join(os.path.dirname(__file__), rel_path)
            spec = importlib.util.spec_from_file_location(mod_name, full_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)


_ensure_lotus_path()


def _get_pipeline(pretrained_model_path: str, task_name: str):
    """Get or create a cached Lotus2Pipeline for the given model+task."""
    cache_key = (pretrained_model_path, task_name)

    if cache_key in _PIPELINES:
        logger.info(f"Reusing cached pipeline [{cache_key}].")
        return _PIPELINES[cache_key]

    logger.info(
        f"Initializing pipeline [{cache_key}]... first run may take a while (model download)."
    )

    # Lazy imports (heavy modules)
    from infer import load_lora_and_lcm_weights  # noqa: F401

    # Device & dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. CPU inference will be slow.")

    weight_dtype = torch.bfloat16  # default bf16 matches infer.py

    # Scheduler + Transformer
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler", num_train_timesteps=10
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_path, subfolder="transformer"
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)

    # Load LoRA adapters + LCM (this auto-downloads from HF if not cached locally)
    transformer, local_continuity_module = load_lora_and_lcm_weights(
        transformer,
        None,  # core_predictor_model_path — let infer.py auto-download
        None,  # lcm_model_path
        None,  # detail_sharpener_model_path
        task_name,
    )

    # Build pipeline with LCM attached
    from pipeline import Lotus2Pipeline

    pipeline = Lotus2Pipeline.from_pretrained(
        pretrained_model_path,
        scheduler=noise_scheduler,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline.set_progress_bar_config(disable=True)

    # Cache and return tuple of (pipeline, device) — we need device for inference calls too
    _PIPELINES[cache_key] = (pipeline.to(device), device)

    logger.info(f"Pipeline [{cache_key}] initialized successfully.")
    return _PIPELINES[cache_key]


class Lotus2Infer:
    """ComfyUI node that wraps Lotus-2 inference pipeline for depth/normal estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pretrained_model_name_or_path": (
                    "STRING",
                    {"default": "black-forest-labs/FLUX.1-dev"},
                ),
                "task_name": (["depth", "normal"],),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/depth_normal_estimation"

    def process(
        self, 
        image, 
        pretrained_model_name_or_path, 
        task_name, 
        num_inference_steps
    ):
        """Run Lotus-2 inference on input image via temp file I/O."""
        
        # Get cached pipeline (downloads + loads models on first call)
        pipeline, device = _get_pipeline(pretrained_model_name_or_path, task_name)

        # Take first batch only
        img_tensor = image[0].cpu()  # [H, W, C] float32
        
        # Convert ComfyUI tensor → PIL Image (uint8 RGB in range [0, 255])
        if img_tensor.min() < 0:
            img_np = ((img_tensor + 1.0) * 0.5).clamp(0, 1).numpy()
        else:
            img_np = img_tensor.clamp(0, 1).numpy()
        pil_image = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")

        # Create temp dir for input file I/O (process_single_image requires a path).
        tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        tmp_file = os.path.join(tmp_dir, f"lotus2_input_{uuid.uuid4().hex}.png")
        try:
            pil_image.save(tmp_file)

            # Import and run inference (uses infer.py's own pipeline path).
            from infer import process_single_image

            _, output_vis_pil, _ = process_single_image(
                tmp_file,
                pipeline,
                task_name=task_name,
                device=device,
                num_inference_steps=num_inference_steps,
            )

            # Convert the visualization image → ComfyUI tensor [1, H, W, C] float32 [0–1].
            vis = output_vis_pil if isinstance(output_vis_pil, Image.Image) else Image.fromarray(np.asarray(output_vis_pil))
            result_tensor = torch.from_numpy(np.array(vis.convert("RGB")).astype(np.float32) / 255.0)[None, ...]

        finally:
            # Clean up temp file
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError as e:
                    logger.warning(f"Failed to cleanup {tmp_file}: {e}")

        return (result_tensor,)
