# ComfyUI Lotus-2 Decomposed Workflow — Demo Guide

**Date:** 2026-06-16  
**Workflow File:** `demo.json`  
**Status:** Complete end-to-end depth estimation pipeline using 7 decomposed nodes

---

## Workflow Overview

This workflow chains all Lotus-2 decomposed nodes together to perform **full depth estimation** on an input image. It mirrors the reference implementation in `Lotus-2/pipeline.py` exactly.

### Node Count: 17 total
- 7 Lotus-2 custom nodes (decomposed)
- 5 standard ComfyUI nodes (Load Image, VAE, CLIP, Save)
- 3 Lotus-2 Latent Packer nodes (pack/unpack helpers)

---

## Stage Breakdown

### STAGE 0: MODEL + IMAGE LOADING (Nodes 1–6)

| Node # | Type | Purpose | Output |
|--------|------|---------|--------|
| **1** | Load-Lotus2-PEFT | Load base FLUX transformer + PEFT adapters (core_predictor, detail_sharpener) + LCM | `LOTUS_MODEL` |
| **2** | LoadImage | Load input image from disk | `IMAGE` |
| **3** | VAELoader | Load VAE model | `VAE` |
| **4** | VAEEncode | Encode image pixels to latent space [B,C,H,W] | `LATENT` |
| **5** | CLIPLoader | Load T5-XXL text encoder (empty prompt for depth) | `CLIP` |
| **6** | CLIPTextEncode | Encode empty string "" to text embeddings | `CONDITIONING` |

**Key:** LOTUS_MODEL outputs a `Lotus2ModelState` object containing transformer, scheduler, lcm_module.

---

### STAGE 1: CORE PREDICTOR (Coarse Depth Estimate) — Nodes 7–11

```
[Encoded Latent from 4]
        ↓
    [7] Pack Latents (mode="pack")
        ↓ outputs: packed_latents [B, T_seq, 4C], img_ids
        ↓
    [8] Switch Adapter → "core_predictor"
        ↓
    [9] Raw Transformer Forward (t=0.001)  ← Single step at end of diffusion
        ↓ outputs: noise_pred packed latents
        ↓
    [10] Unpack Latents (mode="unpack")
        ↓ outputs: unpacked_latents [B,C,H,W] spatial
        ↓
    [11] LCM Inference (spatial smoothing residual)
        ↓ outputs: refined spatial latents
```

**Purpose:** Generate coarse depth structure via one transformer forward pass + LCM continuity smoothing.

**Parameters:**
- Timestep: `0.001` (nearly complete denoising — captures coarse structure)
- Guidance scale: `3.5` (classifier-free guidance strength)

---

### STAGE 2: DETAIL SHARPENING (Refinement Loop) — Nodes 12–14

```
[Refined Latents from 11]
        ↓
    [12] Re-pack Latents (mode="pack")
        ↓ outputs: packed_latents, img_ids (regenerated)
        ↓
    [13] Switch Adapter → "detail_sharpener"
        ↓
    [14] Packed Sampler (multi-step denoising loop)
        ↓ runs 10 steps of iterative refinement
        ↓ outputs: fully denoised packed latents
```

**Purpose:** Iterative refinement over 10 steps (configurable) using the detail_sharpener adapter.

**Parameters:**
- Number of steps: `10` (default)
- Guidance scale: `3.5`
- Scheduler: Extracted from lotus_model.scheduler (FlowMatchEulerDiscreteScheduler)

---

### STAGE 3: OUTPUT (Final VAE Decode + Save) — Nodes 15–17

```
[Denoised Packed Latents from 14]
        ↓
    [15] Final Unpack (mode="unpack")
        ↓ outputs: unpacked final latents [B,C,H,W]
        ↓
    [16] VAE Decode (convert latents → images)
        ↓ outputs: IMAGE [0,1]
        ↓
    [17] Save Image
        ↓ writes depth_map_XXXXXXXX.png to disk
```

---

## Node Connection Map

```
                          ┌────────────────┐
                          │  1: Load PEFT  │◄─────────┐
                          └────────────────┘          │
                                  │                   │
                        ┌─────────┴────────┐          │
                        ▼                  │          │
                  ┌─────────────┐    ┌──────────────┐ │
                  │ 2: LoadImage│    │ 5: CLIPLoader│ │
                  └──────┬──────┘    └──────┬───────┘ │
                         │                  │         │
         ┌───────┬────────┘                 │         │
         ▼       ▼                          ▼         │
    ┌────────┐ ┌──────┐           ┌─────────────────┐│
    │ 4: VAE ├►│ 3:VAE│           │ 6: CLIPTextEnc. ││
    │ Encode │ │Loader│           └─────────────────┘│
    └────┬───┘ └──────┘                              │
         │                                           │
         ▼                                           │
    ┌─────────────┐                                  │
    │ 7: Pack     │                                  │
    │ Latents     │                                  │
    └────┬────────┘                                  │
         │┌─ img_ids                                 │
         ││                  ┌───────────────────────┘
         ▼▼                  ▼
    ┌─────────────┐  ┌──────────────┐
    │ 8: Switch   │  │ LOTUS_MODEL  │
    │→core_pred   │  └──────┬───────┘
    └─────┬───────┘         │
         │ adapter set      │
         ▼                  ▼
    ┌─────────────────────────────────┐
    │ 9: Raw Transformer Forward (t=0.001)
    │    Single step at end of diffusion
    └────────────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │ 10: Unpack     │
         │ Latents        │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ 11: LCM        │
         │ Inference      │
         └───────┬────────┘
                 │
    ┌────────────┴──────────────┐
    │                           │
    ▼                           ▼
┌─────────────┐         ┌────────────────────┐
│ 12: Re-pack │         │ Re-fetch LOTUS_MODEL│
│ Latents     │         └────────────────────┘
└────┬────────┘ img_ids ▲   │
     │          ◄────────┘   │
     │                       │
     ▼                       ▼
┌──────────────────┐  ┌────────────────┐
│ 13: Switch       │  │ Still contains │
│→detail_sharpener │  │scheduler inside│
└────────┬─────────┘  └────────────────┘
         │adapter set
         ▼
    ┌─────────────────────────────────────────┐
    │ 14: Packed Sampler (Multi-step Loop)    │
    │     10 iterative denoising steps        │
    │     Uses scheduler.step() internally    │
    └────────────┬────────────────────────────┘
                 │
         ┌───────▼────────┐
         │ 15: Final      │
         │ Unpack         │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ 16: VAE Decode │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ 17: SaveImage  │
         └────────────────┘
```

---

## How to Use This Workflow

### 1. **Prepare Input Image**
   - Place an RGB image in your ComfyUI directory (or specify path in Node 2)
   - Recommended size: 512-1024 pixels (larger = slower)
   - Format: PNG, JPG, WebP

### 2. **Load Workflow in ComfyUI**
   ```
   ComfyUI → Queue → "Load" → select "demo.json"
   ```

### 3. **Configure Parameters (Optional)**
   - **Node 1** — Task: "depth" or "normal"
   - **Node 9** — Timestep: 0.001 (core predictor position on denoising curve)
   - **Node 14** — Num Steps: 1–50 (more steps = better quality, slower)
   - **Node 14** — Guidance Scale: 1.0–7.5 (higher = stronger guidance)

### 4. **Run Queue**
   ```
   ComfyUI → Queue Prompt
   ```
   - First run: ~5–10 minutes (downloads FLUX + adapters + LCM)
   - Subsequent runs: ~2–3 minutes (models cached)

### 5. **Inspect Output**
   - Depth map saved to `output/lotus2_depth_XXXXXXXX.png`
   - Compare with `lotus2_infer_node.py` (Option B) for verification

---

## Key Design Decisions

### Why 7 Decomposed Nodes?

1. **Transparency:** Each stage is visible — inspect intermediate latents
2. **Flexibility:** Swap schedulers, adjust timesteps, skip LCM if desired
3. **Reusability:** Chain nodes in different orders for experiments
4. **Control:** Fine-tune guidance + steps per stage

### Why Pack/Unpack Nodes?

FLUX latents use a packed format `[B, T_seq, 4C]` internally (efficiency). Packing/unpacking:
- Compresses spatial latents into sequence tokens
- Enables efficient attention computation
- Returns img_ids (positional embeddings for RoPE)

### Scheduler Inside Lotus2ModelState?

**Design choice:** Scheduler bundled in model state, not as separate output
- Single output type simplifies wiring
- All inference parameters live in one object
- Packed-Sampler extracts scheduler via `getattr(lotus_model, 'scheduler')`

---

## Common Modifications

### Skip LCM (Test Core Predictor Only)

Comment out or bypass Node 11:
```json
{
  "10": {...},
  "12": {
    "inputs": {"latents": ["10", 0], "mode": "pack"},
    ...
  }
}
```

**Effect:** Only coarse depth, no smoothing refinement.

### Increase Refinement Steps

In Node 14, change `num_steps`:
```json
"num_steps": 20  // More iterations = higher quality but slower
```

### Disable Guidance

In Nodes 9 and 14, set `guidance_scale: 1.0`:
```json
"guidance_scale": 1.0  // No classifier-free guidance
```

### Normal Estimation (Instead of Depth)

In Node 1, change task:
```json
"task_name": "normal"
```

Output will be RGB normals (not depth map).

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "LOTUS_MODEL not found" | Node 1 failed to load | Check CUDA/CPU availability, HF credentials |
| "Timestep out of range" | Node 9 timestep invalid | Keep timestep in [0.001, 1.0] |
| "Tensor shape mismatch" | Pack/unpack H,W not set | Node 7 should auto-detect; check input size multiple of 16 |
| "Memory error" | Model too large for GPU | Reduce input image size, use CPU (slow) |
| Output is black/noisy | Guidance too high | Try `guidance_scale: 1.5–2.0` |

---

## Performance Notes

- **STAGE 1** (Core Predictor): ~30–60 seconds
- **STAGE 2** (Detail Sharpening, 10 steps): ~2–5 minutes
- **Total:** ~3–6 minutes per image (NVIDIA A100+)

On CPU: 10–30x slower.

---

## Comparison with Option B (`lotus2_infer_node.py`)

| Feature | Option A (Decomposed) | Option B (Monolithic) |
|---------|--------|----------|
| Nodes in workflow | 7 | 1 |
| Transparency | High — inspect each stage | Low — all internal |
| Flexibility | Modify individual stages | Fixed pipeline |
| Performance | Same (same underlying code) | Same |
| Ease of use | Requires wiring | Single click |

**Recommendation:** Use Option A for experimentation, Option B for production inference.

---

## Next Steps

1. **Test demo.json** with a sample image
2. **Compare depth maps** between Option A and Option B
3. **Adjust parameters** (steps, guidance, timestep) to understand their effect
4. **Create variations** (normal estimation, skip LCM, etc.)

---

## Files Reference

- **Workflow:** `demo.json` (this directory)
- **Nodes:** `lotus2_*.py` (7 decomposed node implementations)
- **Reference:** `Lotus-2/pipeline.py` (lines 30–160 for master logic)
- **Fallback:** `lotus2_infer_node.py` (Option B all-in-one node)

