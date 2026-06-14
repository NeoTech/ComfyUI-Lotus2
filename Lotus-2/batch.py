#!/usr/bin/env python
# coding=utf-8
"""
Lotus-2 Batch Inference Script

Reads a list of image paths (from file or directory), runs inference,
and saves output with "_lotus" appended to each filename.

Usage:
    # Auto-download models, process all images in a directory:
    python batch.py --input_dir ./images

    # Use a list file with local model paths:
    python batch.py --list_file list.txt --core_predictor_model_path ./weights/core.safetensors

    # Process normals instead of depth:
    python batch.py --input_dir ./images --task_name normal
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from infer import load_lora_and_lcm_weights
from pipeline import Lotus2Pipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run Lotus-2 batch inference.")
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Path to a single image file to process. "
             "Mutually exclusive with --input_dir and --list_file.",
    )
    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Path to a text file containing one image path per line. "
             "Provide exactly one of: --single-file, --list_file, or --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input images (auto-discovers .png/.jpg). "
             "Provide exactly one of: --single-file, --list_file, or --input_dir.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--core_predictor_model_path",
        type=str,
        default=None,
        help="Path to core predictor model weights",
    )
    parser.add_argument(
        "--lcm_model_path",
        type=str,
        default=None,
        help="Path to local continuity module model weights",
    )
    parser.add_argument(
        "--detail_sharpener_model_path",
        type=str,
        default=None,
        help="Path to detail sharpener model weights",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files, e.g. fp16",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of timesteps to infer the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. (default: 42)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth",
        choices=["depth", "normal"],
        help="Task name: depth or normal.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to process per GPU forward pass. (default: 8)",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def load_image_list(list_file):
    """Load image paths from a text file (one per line)."""
    with open(list_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def discover_images(input_dir):
    """Auto-discover .png and .jpg images in a directory."""
    input_dir = Path(input_dir)
    images = sorted(list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")))
    return images


def get_image_paths(args):
    """Get image paths from --single-file, --list_file, or --input_dir."""
    provided = sum(1 for v in (args.single_file, args.list_file, args.input_dir) if v is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of: --single-file, --list_file, or --input_dir.")

    if args.single_file is not None:
        path = Path(args.single_file)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {args.single_file}")
        return [path]
    elif args.list_file is not None:
        return load_image_list(args.list_file)
    else:
        return discover_images(args.input_dir)


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stderr,
        force=True,
    )

    image_paths = get_image_paths(args)
    logging.info(f"Found {len(image_paths)} images to process.")

    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directory (vis only — no .npy)
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_vis = os.path.join(args.output_dir, f"{args.task_name}_vis")
    os.makedirs(output_dir_vis, exist_ok=True)
    logging.info(f"Output dir = {output_dir_vis}")

    # Mixed precision
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    logging.info(f"Running with {weight_dtype} precision.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Load models --------------------
    logging.info(f"[1/6] Loading noise scheduler from {args.pretrained_model_name_or_path} ...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", num_train_timesteps=10
    )
    logging.info("  Scheduler loaded.")

    logging.info("[2/6] Loading Flux transformer (this may take a minute) ...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)
    logging.info("  Transformer loaded.")

    logging.info("[3/6] Loading LoRA & LCM weights (core_predictor, lcm, detail_sharpener) ...")
    transformer, local_continuity_module = load_lora_and_lcm_weights(
        transformer,
        args.core_predictor_model_path,
        args.lcm_model_path,
        args.detail_sharpener_model_path,
        args.task_name,
    )
    logging.info("  LoRA & LCM weights loaded.")

    logging.info("[4/6] Building pipeline ...")
    pipeline = Lotus2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    logging.info("  Pipeline ready.")

    # -------------------- Run inference (batched) --------------------
    logging.info(f"[5/6] Processing {len(image_paths)} images (batch_size={args.batch_size}) ...")

    processed = 0
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    for batch_start in range(0, len(image_paths), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        batch_num = (batch_start // args.batch_size) + 1

        logging.info(f"  Batch [{batch_num}/{total_batches}] ({len(batch_paths)} images)")

        # Load & stack into [B, C, H, W] — pad to largest in batch
        tensors = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image).astype(np.float32)
            image_ts = torch.tensor(image_np).permute(2, 0, 1)  # [C, H, W]
            tensors.append(image_ts)

        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)

        padded = []
        for t in tensors:
            h, w = t.shape[1], t.shape[2]
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h or pad_w:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0)
            padded.append(t)

        rgb_batch = torch.stack(padded) / 127.5 - 1.0  # [B, C, H, W]
        rgb_batch = rgb_batch.to(device)

        max_edge = max(max_h, max_w)
        process_res = 1024 if max_edge > 1024 else (512 if max_edge < 512 else None)

        predictions = pipeline(
            rgb_in=rgb_batch,
            prompt="",
            num_inference_steps=args.num_inference_steps,
            output_type="np",
            process_res=process_res,
        ).images

        for image_path, prediction in zip(batch_paths, predictions):
            if args.task_name == "depth":
                output_npy = prediction.mean(axis=-1)
                output_vis = colorize_depth_map(output_npy, reverse_color=True)
            elif args.task_name == "normal":
                output_vis = Image.fromarray((prediction * 255).astype(np.uint8))

            stem = Path(image_path).stem
            output_vis.save(os.path.join(output_dir_vis, f"{stem}_lotus.png"))

        processed += len(batch_paths)
        logging.info(f"  Saved {processed}/{len(image_paths)} images")

    logging.info(f"[6/6] Done. Processed {processed} images.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
