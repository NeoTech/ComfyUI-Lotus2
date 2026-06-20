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

    @staticmethod
    def _force_active_adapter(transformer, adapter_name):
        """Force PEFT active adapter state for transformer and nested LoRA modules."""
        if not hasattr(transformer, "peft_config") or transformer.peft_config is None:
            raise RuntimeError(
                "No PEFT adapters found on the transformer — did you skip Lotus2PeftLoader? "
                "`transformer.peft_config` is missing."
            )

        available = set(transformer.peft_config.keys())
        if adapter_name not in available:
            raise RuntimeError(
                f"Adapter '{adapter_name}' is not loaded on the transformer. "
                f"Available adapters: {sorted(available)}."
            )

        # Ask PEFT to switch first.
        try:
            transformer.set_adapter(adapter_name)
        except Exception as e:
            logger.warning(
                "PEFT set_adapter(%s) failed (%s); continuing with forced active state.",
                adapter_name,
                e,
            )

        # Force top-level PEFT bookkeeping.
        if hasattr(transformer, "_active_adapter"):
            try:
                transformer._active_adapter = adapter_name
            except Exception as e:
                logger.warning(
                    "Could not force transformer._active_adapter=%s (%s).",
                    adapter_name,
                    e,
                )

        # Also clear any stale active state that PEFT may have restored.
        if hasattr(transformer, "_active_adapters"):
            try:
                transformer._active_adapters = [adapter_name]
            except Exception as e:
                logger.warning(
                    "Could not force transformer._active_adapters=%s (%s).",
                    adapter_name,
                    e,
                )

        if hasattr(transformer, "active_adapter"):
            try:
                transformer.active_adapter = adapter_name
            except Exception as e:
                logger.warning(
                    "Could not force transformer.active_adapter=%s (%s).",
                    adapter_name,
                    e,
                )

        # Some PEFT/diffusers versions expose active state through _active_adapters.
        if hasattr(transformer, "_active_adapters"):
            try:
                transformer._active_adapters = [adapter_name]
            except Exception as e:
                logger.warning(
                    "Could not force transformer._active_adapters=%s (%s).",
                    adapter_name,
                    e,
                )

        if hasattr(transformer, "active_adapter"):
            try:
                transformer.active_adapter = adapter_name
            except Exception as e:
                logger.warning(
                    "Could not force transformer.active_adapter=%s (%s).",
                    adapter_name,
                    e,
                )
                    

        # Force nested LoRA/PEFT modules too. This matters because some forward paths
        # may consult module-level active adapter state instead of only the top-level attr.
        for module in transformer.modules():
            set_adapter = getattr(module, "set_adapter", None)
            if callable(set_adapter):
                try:
                    set_adapter(adapter_name)
                except Exception as e:
                    logger.debug(
                        "Ignoring recursive set_adapter failure on %s: %s",
                        type(module).__name__,
                        e,
                    )

            for attr in ("_active_adapter", "active_adapter"):
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, adapter_name)
                    except Exception as e:
                        logger.debug(
                            "Could not force %s.%s=%s (%s)",
                            type(module).__name__,
                            attr,
                            adapter_name,
                            e,
                        )

            if hasattr(module, "_active_adapters"):
                try:
                    module._active_adapters = [adapter_name]
                except Exception as e:
                    logger.debug(
                        "Could not force %s._active_adapters=%s (%s)",
                        type(module).__name__,
                        adapter_name,
                        e,
                    )

            # PEFT versions can expose _active_adapter as None even after set_adapter() succeeds.
            # The workflow-level source of truth is now lotus_model.active_adapter.
            # Log once after all modules have been updated.
            actual_active = getattr(transformer, "_active_adapter", None)
            if isinstance(actual_active, (list, tuple)):
                actual_active = list(actual_active)

            if actual_active != adapter_name:
                logger.warning(
                    "PEFT did not expose active adapter as '%s' after forcing; continuing with forced state. "
                    "Actual PEFT value is %s.",
                    adapter_name,
                    actual_active,
                )


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
        current_active = getattr(lotus_model.transformer, "_active_adapter", None)
        logger.info(
            f"Lotus2: Adapter {'loaded' if current_active != adapter_name else 'reusing cache'} - mode={adapter_name}"
        )

        self._force_active_adapter(lotus_model.transformer, adapter_name)

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