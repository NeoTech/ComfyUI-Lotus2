# ComfyUI-Lotus2 Development Handover

**Last Updated:** 2026-06-16 (cache management + explicit cleanup node)
**Status:** 95% code-complete; awaiting demo workflow + integration testing
---
## Recent Changes

### PEFT Loader Cache Management (2026-06-16)
Added reference-counted caching to `lotus2_peft_loader.py`:
- **Lotus2ModelState** now tracks `_ref_count` per cached entry
- **load()** increments ref count on cache hit, sets 1 on new creation
- **_cleanup_model_state()** helper decrements by 1; unloads to CPU + clears from cache when hits 0
- **New Lotus2ModelCleanup node**: Users can add "Lotus2: Release Model Memory" to workflow to explicitly free VRAM
    
**IMPORTANT CONSTRAINT:** ComfyUI has no `on_node_disconnect` hook for custom nodes. The ref count must be manually decremented by calling the cleanup node — it will NOT auto-unload when you disconnect loader outputs in the UI.

---

## Implementation Overview

**Reference Scripts** (standalone implementations):
- `Lotus-2/infer.py` — Standalone inference entry point
- `Lotus-2/pipeline.py` — **Lotus2Pipeline class** — **PRIMARY REFERENCE for decomposed node logic**

**ComfyUI Implementations:**
- `lotus2_infer_node.py` — **Simple wrapper** around infer.py + pipeline.py (Option B fallback)
- 7 decomposed nodes — Replicate `pipeline.py` logic stage-by-stage (Option A primary)

**To understand node logic:** Read `Lotus-2/pipeline.py:Lotus2Pipeline.__call__()` (lines 30–160)

---

## Quick Status

✅ **COMPLETE:**
- All 7 ComfyUI nodes created and registered
- Lotus2ModelState dataclass for cross-node state transfer
- Adapter switching, latent packing/unpacking, LCM inference, denoising sampler
- Integration with standard ComfyUI VAE/CLIP nodes

⚠️ **CRITICAL BLOCKER:**
- **No functional demo workflow** — users cannot see how to wire the nodes together
- No end-to-end integration testing completed
- No user-facing documentation on workflow composition

---

## Immediate Next Steps

### 1. Create Demo Workflow (HIGHEST PRIORITY)
Create `demo_workflow_depth.json` showing complete node chain:
```
Load-PEFT → [VAE Encode] → Pack → [Switch→core] → [Raw-Forward] → [Unpack] → 
[LCM] → [Pack] → [Switch→detail] → [Packed-Sampler] → [Unpack] → [VAE Decode]
```
**Deliverables:** JSON file, PNG screenshot, markdown wiring guide

### 2. Integration Testing
- Test all node imports in ComfyUI
- Verify Lotus2ModelState passes through chain correctly
- Run end-to-end inference and compare output to `lotus2_infer_node.py` baseline
- Test pack/unpack roundtrip for numerical stability

### 3. User Documentation
- Node API reference (inputs/outputs/types)
- Workflow assembly guide (step-by-step)
- Troubleshooting common issues

---

## File Inventory

| File | Purpose | Status | Reference |
|------|---------|--------|-----------|
| `Lotus-2/infer.py` | Standalone inference + model loading | ✅ | Source: `load_lora_and_lcm_weights()`, `process_single_image()` |
| `Lotus-2/pipeline.py` | Lotus2Pipeline class (extended FluxPipeline) | ✅ | **PRIMARY: Read `__call__()` lines 30–160 for node decomposition** |
| `lotus2_infer_node.py` | Option B: wrapper node (simple wrapper) | ✅ | Wraps infer.py + pipeline.py |
| `lotus2_utils.py` | Shared LCM class, pack/unpack wrappers | ✅ | Extracted from infer.py + diffusers |
| `lotus2_peft_loader.py` | Load model + adapters (scheduler bundled in state) | ✅ | Based on infer.py:load_lora_and_lcm_weights(); outputs single Lotus2ModelState |
| `lotus2_lcm_inference.py` | LCM spatial smoothing node | ✅ | Based on pipeline.py lines 94–95 |
| `lotus2_adapter_switcher.py` | Toggle adapters (core_pred ↔ detail_sharp) | ✅ | Based on pipeline.py lines 84, 96 (set_adapter calls) |
| `lotus2_latent_packer.py` | Pack/unpack latent tensors | ✅ | Based on pipeline.py lines 52–59 (pack/unpack calls) |
| `lotus2_raw_transformer_forward.py` | Single-step transformer inference | ✅ | Based on pipeline.py lines 87–93 (core predictor forward) |
| `lotus2_packed_sampler.py` | Denoising loop with sigmas | ✅ | Based on pipeline.py lines 101–126 (detail sharpener loop) |
| `__init__.py` | Node registration | ✅ | Registers all 7 decomposed nodes + fallback |

---

## Known Constraints

- **Lotus2ModelState bundles all state:** Transformer, LCM, scheduler, task_name, device are all in ONE object. No separate scheduler output—accessed via `getattr(lotus_model, 'scheduler')` in Packed-Sampler.
- **Adapter switching is stateful:** `set_adapter()` mutates model in-place; parallel branches will conflict
- **No error validation:** Limited type checking at node boundaries
- **Device placement:** All parameters must be on same device; LCM requires explicit `.to(device)` handling
- **Timestep bounds:** Raw-Forward accepts [0.001, 1.0]; edge cases at t=0 untested
- **Cache cleanup is manual:** Users MUST add "Lotus2: Release Model Memory" node to free VRAM after use. Disconnecting loader outputs does NOT trigger auto-unload.

---

## How to Verify Integration

1. Start ComfyUI
2. Load `demo_workflow_depth.json` (once created)
3. Select test image
4. Run workflow
5. Compare output depth map to `lotus2_infer_node.py` on same input
6. Should be visually identical (allow ±0.5% numerical tolerance)

---

## Quick Links

- **Reference impl:** `Lotus-2/infer.py`, `Lotus-2/pipeline.py`
- **Shared state:** `Lotus2ModelState` dataclass in `lotus2_peft_loader.py`
- **Node wiring diagram:** To be created in demo workflow JSON
- **Full analysis:** See `ComfyUI_Lotus2_HANDOFF.md` in system temp directory

---

## For Next Agent

Suggested starting points:
1. Read `ComfyUI_Lotus2_HANDOFF.md` for architectural overview
2. Create demo workflow (priority 1)
3. Run integration tests
4. Write user documentation
5. Mark all verification checkboxes in `todo.md` (if file is present)

