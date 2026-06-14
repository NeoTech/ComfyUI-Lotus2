import argparse
import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def convert_depth_perfect_16bit(image_path, output_path):
    """Convert a single color depth map to 16-bit grayscale PNG.

    Returns (image_path, success: bool) for progress tracking.
    """
    try:
        # 1. Load color depth map image safely
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return (image_path, False)

        # 2. Convert to 64-bit float space for smooth, lossless precision math
        b = img[:, :, 0].astype(np.float64) / 255.0
        g = img[:, :, 1].astype(np.float64) / 255.0
        r = img[:, :, 2].astype(np.float64) / 255.0

        # 3. Apply Relative Channel Vectoring
        raw_depth = r - b
        depth_carved = raw_depth - (g * 0.35)

        # 4. Normalize to 16-bit range (0 - 65535)
        depth_normalized = (depth_carved - np.min(depth_carved)) / (
            np.max(depth_carved) - np.min(depth_carved)
        )
        final_depth = (depth_normalized * 65535.0).astype(np.uint16)

        # 5. Clean up micro AI noise
        final_depth = cv2.GaussianBlur(final_depth, (3, 3), 0)

        # 6. Save as single-channel high-fidelity 16-bit PNG
        cv2.imwrite(output_path, final_depth)
        return (image_path, True)
    except Exception:
        return (image_path, False)


def _convert_worker(args_tuple):
    """Worker function for ProcessPoolExecutor."""
    image_path, output_path = args_tuple
    return convert_depth_perfect_16bit(image_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert color depth maps to 16-bit grayscale PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python conversion.py --input_dir ./depth_vis --output_dir ./converted\n"
            "  python conversion.py --list_file list.txt --output_dir ./converted\n"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input images (auto-discovers .png/.jpg).",
    )
    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Path to a text file containing one image path per line.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./converted",
        help="Output directory for converted 16-bit PNGs. (default: ./converted)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to use (1 = sequential). (default: 1)",
    )
    return parser.parse_args()


def discover_images(input_dir):
    """Auto-discover .png and .jpg images in a directory."""
    input_dir = Path(input_dir)
    images = sorted(list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")))
    return images


def load_image_list(list_file):
    """Load image paths from a text file (one per line)."""
    with open(list_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_image_paths(args):
    if args.input_dir is not None:
        return discover_images(args.input_dir)
    elif args.list_file is not None:
        return load_image_list(args.list_file)
    else:
        raise ValueError("Provide either --input_dir or --list_file.")


def main():
    args = parse_args()
    image_paths = get_image_paths(args)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(image_paths)} images. Output dir: {args.output_dir}")
    print(f"Cores: {args.cores} {'(parallel)' if args.cores > 1 else '(sequential)'}")

    tasks = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        output_path = os.path.join(args.output_dir, f"{stem}_conversion.png")
        tasks.append((image_path, output_path))

    failed = 0
    if args.cores > 1:
        with ProcessPoolExecutor(max_workers=args.cores) as executor:
            futures = {executor.submit(_convert_worker, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Converting"):
                image_path, success = future.result()
                if not success:
                    failed += 1
    else:
        for idx, task in enumerate(tqdm(tasks, desc="Converting"), 1):
            image_path, output_path = task
            _, success = _convert_worker(task)
            if not success:
                failed += 1

    print(f"Done. Converted {len(image_paths) - failed}/{len(image_paths)} images.")
    if failed:
        print(f"  Failed: {failed} image(s)")


if __name__ == "__main__":
    main()