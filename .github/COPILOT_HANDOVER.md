# ComfyUI-Lotus2 Handover — Direct Block Inference Implementation

## Objective

Bypass `comfy.patcher_extension.WrapperExecutor` 4D-shape assertion (`bs, c, h_orig, w_orig = x.shape`) that crashes when passing packed sequence tensors `[B,T,4C]` instead of spatial latents `[B,C,H,W]`. Solution: extract the raw Flux nn.Module from ModelPatcher and call internal methods (process_img, double_blocks[i], single_blocks[i], final_layer) directly — pure PyTorch with no wrapper interception.

## Problem Origin
- ComfyUI's `Flux.forward()` is wrapped by WrapperExecutor via `_forward()`, which expects `[B,C,H,W]` spatial tensors.
- Lotus-2 pipeline needs packed sequence format `[B,T,4C]` for the core predictor step (t=1.0) and detail sharpener loop.
- Passing packed sequences to `model.model()` hits WrapperExecutor → shape crash.

## Architecture Discovery
### Model Stack (Confirmed via GitHub source: comfy/ldm/flux/model.py)
```
ModelPatcher.__dict__["model"]  # DiffusionModel wrapper (has apply_model, encode_adm, model_sampling...)
    └── .diffusion_model         # Flux(nn.Module) — actual transformer with time_in, double_blocks, etc.
        ├── img_in               # Linear(in_channels*patch**2 → hidden_size)
        ├── txt_in               # Linear(context_in_dim → hidden_size)  
        ├── pe_embedder          # EmbedND(dim=pe_dim, theta=params.theta, axes_dim=[...])
        ├── time_in              # MLPEmbedder(256 → hidden_size)
        ├── guidance_in          # MLPEmbedder or nn.Identity (depends on params.guidance_embed)
        ├── vector_in            # MLPEmbedder(params.vec_in_dim, hidden_size) or None
        ├── double_blocks        # ModuleList[DoubleStreamBlock] × depth
        ├── single_blocks        # ModuleList[SingleStreamBlock] × depth_single_blocks  
        ├── final_layer          # LastLayer(hidden_size → out_channels_flat)
```

### Key: forward_orig() Call Pattern (from comfy/ldm/flux/model.py)
1. `img, img_ids = self.process_img(x)` — spatial [B,C,H,W] → packed [B,T_img,H_dim], positional IDs
2. `vec = self.time_in(timestep_embedding(...))` + guidance/vector modulation
3. `txt = self.txt_in(context)` 
4. RoPE: `pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))`
5. Double blocks loop: `img_out, txt_out = block(img=img, txt=txt, vec=vec, pe=pe, transformer_options={})`
6. Merge + single blocks: `combined = torch.cat((txt, img), dim=1)` → iterate single_blocks[i](combined, vec=vec, pe=pe)
7. Final layer: `final_layer(combined[:, txt_len:], vec_orig)` — image tokens only

### WrapperExecutor Behavior
- Only wraps method calls INSIDE `Flux.forward()` → `_forward()`. Direct attribute reads and calling sub-modules (`block(...)`, `final_layer(...)`) are pure PyTorch with zero interception.

## File State (`lotus2_inference.py`) — 2026-06-14 Fixes

### Applied (all bugfixes stable, latest log: no crash):

| # | Fix | Before | After | Status |
|---|-----|--------|-------|--------|
| A1 | Crash line tuple-unpack | `t0,t1 = latents[...].float()` — ValueError | `t_sample = ...` single assign | ✅ Stable |
| B1-B3 | Core predictor rewrite | Gaussian noise at t=1.0, broken flow formula | VAE-encoded RGB input at t≈0.001, direct output | ✅ Stable |
| C1-C2 | Euler scheduler.step + /1000 scaling | Manual sigma math, raw timesteps | `scheduler.step()` with `(t/1000)` | ✅ Stable |
| D3-E2 | h_len floor div, guidance float32 | Ceiling division, bf16 dtype | Floor div, float32 per diffusers conv. | ✅ Stable |
| Input norm `[0,1]→[-1,+1]` | Raw ComfyUI IMAGE [0,1] | `rgb_in * 2 - 1` before VAE encode | ✅ Stable (mean now ~-0.03) | 
| **Float32 Euler loop** | Mixed dtype: noise_pred float32 + latents bf16 in step() → truncation drift × 10 steps = off-scale `[-2.7,+1.8]` | Both cast to `.float()` before scheduler.step(), restore after loop | 🔴 **NEW — test pending** |
| **Prompt padding to 512 + use user prompt param** | Hardcoded `clip.tokenize("")`, short txt_len → wrong img_part extraction every step | Use `[prompt] if given else [""]` + zero-pad to 512 tokens matching diffusers encode_prompt | 🔴 **NEW — test pending** |

### Remaining:
- Test with real image after float32 Euler loop + prompt padding fixes. Verify decoded pre-clamp range is within `[-1.5, +1.5]` (was [-2.7,+1.8]). Pixel sample should show varied values instead of all 1.0.

## Key Files
| File | Status | Notes |
|------|--------|-------|
| `lotus2_inference.py` | CLEAN ✓ (367 lines) | _get_raw_flux fixed to diffusion_model, _flux_forward complete rewrite using process_img/block/pe_embedder/final_layer pattern from ComfyUI source |
| `lotus2_loader.py` | CLEAN ✓ | LoadLotus2Adapters, LocalContinuityModule working |  
| `__init__.py` | CLEAN ✓ | Node registration correct (3 nodes: Lotus-2 Infer + 2 depth_tools) |
| `depth_tools.py` | NEW ✓ (104 lines) | DepthToAlphaMask + DepthBlendComposite — both accept DEPTHS type directly |
| `tests/test_depth_tools.py` | NEW ✓ | 8 unit tests, all passing with [B,H,W] mock data |

## _flux_forward() Implementation Details (Current)
- Calls Flux.process_img(x_spatial) internally — handles spatial→sequence packing + img_ids generation identically to ComfyUI's path.
- Builds vec = time_in(timestep_emb) + guidance_in(guidance_emb) [if enabled] + vector_in(pooled_vec).
- txt = txt_in(context) or dummy zeros for empty text path.
- txt_ids: zero tensor with optional linspace on axes_dim indices (per flux_module.params.txt_ids_dims).
- RoPE: pe_embedder(cat(txt_ids, img_ids)).
- Double blocks loop calls `block(img=img, txt=txt, vec=vec, pe=pe, transformer_options={})` — returns `(img_out, txt_out)`. Handles global_modulation tuple unpacking if params.global_modulation=True.
- Single blocks: cat((txt,img),dim=1) → iterate single_blocks[i](combined, vec=..., pe=pe). 
- Final layer: `final_layer(combined[:, txt_len:], vec_orig)` — extracts image tokens only, projects to out_channels_flat.
- Output rearrange via einops: `"b (h w) (c ph pw) -> b c (h ph) (w pw)"` with h_len/w_len calculated from patch_size matching process_img's internal grid.

## VAE Constants
```python
VAE_SCALE_FACTOR = 8          # spatial downscale from pixel space  
VAE_SHIFT_FACTOR = 0.0609     # ComfyUI FLUX latent centering offset
VAE_SCALING_FACTOR = 0.3611   # per-channel std normalization multiplier

# Encode: (raw_latents - SHIFT) * SCALE
# Decode: latents / SCALE + SHIFT → .decode() → clamp(0,1)
```

## Do NOT Revert To
- Calling `model.model()` with packed tensors — hits WrapperExecutor and crashes.
- Using custom pack/unpack helpers directly on Flux (use process_img internally).
- Sequential edit attempts using replace_string_in_file on corrupted sections — use python script or atomic file rewrite instead.
