"""ComfyUI node: Switch active PEFT adapter on transformer via set_adapter().

Switches between "core_predictor" and "detail_sharpener" adapters that were loaded
by Lotus2PeftLoader. Because PEFT's set_adapter() is stateful and modifies the
transformer in-place, this acts as an 'enter' node for a branch — downstream nodes
will use whichever adapter was activated last on that transformer instance."""

import logging

logger = logging.getLogger(__name__)


def _get_available_adapters(transformer) -> set:
    """Return the set of PEFT adapter names currently loaded on *transformer*.

    Inspects ``transformer.peft_config`` (dict-like; keys are adapter names).
    Raises RuntimeError when peft_config is missing or empty.
    """
    if not hasattr(transformer, "peft_config") or transformer.peft_config is None:
        raise RuntimeError(
            "No PEFT adapters found on the transformer — did you skip Lotus2PeftLoader? "
            "`transformer.peft_config` is missing."
        )

    available = set(transformer.peft_config.keys())
    if not available:
        raise RuntimeError(
            f"No PEFT adapters loaded. `peft_config` keys are empty: {list(transformer.peft_config.keys())}"
        )

    return available


class Lotus2AdapterSwitcher:
    """Set the active PEFT adapter name on the shared FluxTransformer2DModel.

    Pipeline origin (from Lotus-2/pipeline.py):
        - Line ~137: self.transformer.set_adapter("core_predictor") — Stage 1 single forward pass
        - Line ~154: self.transformer.set_adapter("detail_sharpener") — Stage 2 denoising loop

    WARNING: set_adapter() is stateful and in-place. The transformer retains the last
    adapter name that was set on it. Do NOT share LOTUS_MODEL between parallel workflow
    branches that need different adapters simultaneously — whichever branch executes its
    switch() last will overwrite the other's active adapter."""

    ADAPTERS = ("core_predictor", "detail_sharpener")

    # One-time DAG-branch-safety warning flag
    _DAG_WARNING_SHOWN: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_model": ("LOTUS_MODEL",),
                "adapter_name": (cls.ADAPTERS,),
            }
        }

    RETURN_TYPES = ("LOTUS_MODEL",)
    RETURN_NAMES = ("lotus_model",)
    FUNCTION = "switch"
    CATEGORY = "image/lotus2_decomposed"

    @staticmethod
    def _get_available_adapters(transformer) -> set:
        """Delegate to the module-level helper (keeps class interface tidy)."""
        return _get_available_adapters(transformer)

    def switch(self, lotus_model, adapter_name: str):
        """Switch the active PEFT adapter on the transformer.

        Args:
            lotus_model: Lotus2ModelState carrying the FluxTransformer2DModel with adapters attached.
            adapter_name: One of ("core_predictor", "detail_sharpener").

        Returns:
            Tuple containing the (mutated) LOTUS_MODEL_SWITCHED state object.
        """
        # 1. Validate transformer attribute exists
        if not hasattr(lotus_model, "transformer") or lotus_model.transformer is None:
            raise AttributeError(
                "`lotus_model` has no .transformer — did you pass the output of Lotus2PeftLoader?"
            )

        # 2. Inspect loaded adapters & validate request
        available = self._get_available_adapters(lotus_model.transformer)

        if adapter_name not in available:
            raise RuntimeError(
                f"Adapter '{adapter_name}' is not loaded on the transformer. "
                f"Available adapters: {sorted(available)}. "
                f"Make sure Lotus2PeftLoader was run before this node."
            )

        # 3. Switch adapter (stateful, in-place — mirrors pipeline.py line ~137/~154)
        logger.info("Switching active adapter to: %s", adapter_name)
        lotus_model.transformer.set_adapter(adapter_name)

        # 4. Track current adapter on the state object (dynamic attribute is fine for dataclass instances)
        lotus_model.active_adapter = adapter_name

        # 5. One-time DAG-branch-safety warning
        if not self._DAG_WARNING_SHOWN:
            logger.warning(
                "set_adapter() modifies the transformer in-place — do NOT share LOTUS_MODEL "
                "between parallel workflow branches that need different adapters simultaneously."
            )
            self._DAG_WARNING_SHOWN = True

        return (lotus_model,)