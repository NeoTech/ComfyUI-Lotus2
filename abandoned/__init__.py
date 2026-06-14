from .lotus2_loader import LoadLotus2Adapters, LoadLotus2LCM
from .lotus2_inference import Lotus2InferenceModular

NODE_CLASS_MAPPINGS = {
    "LoadLotus2Adapters": LoadLotus2Adapters,
    "LoadLotus2LCM": LoadLotus2LCM,
    "Lotus2InferenceModular": Lotus2InferenceModular,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLotus2Adapters": "Load Lotus-2 Adapters",
    "LoadLotus2LCM": "Load Lotus-2 LCM Module",
    "Lotus2InferenceModular": "Lotus-2 Inference (Modular)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]