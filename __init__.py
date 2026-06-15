"""ComfyUI-Lotus2 custom node package.

Registers the Lotus-2 Infer node with ComfyUI's custom-node system.
Wraps infer.py + pipeline.py via temporary file I/O for simplicity.
"""

from .lotus2_infer_node import Lotus2Infer

NODE_CLASS_MAPPINGS = {
    "Lotus-2 Infer": Lotus2Infer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lotus-2 Infer": "Lotus2: Infer"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

