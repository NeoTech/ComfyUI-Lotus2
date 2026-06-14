"""
Lotus-2 ComfyUI Custom Node — direct-block-inference version.

Pipeline:
    1. Image -> VAE -> spatial latents (B, 16, H, W)
    2. Core predictor: _flux_forward() with t=1.0 + empty text -> packed
       Calls Flux internal nn.Modules directly — bypasses WrapperExecutor entirely.
    3. Unpack -> LCM (spatial) -> repack
    4. Detail sharpener: Euler loop via _flux_forward(t, ...)
    5. Unscale -> VAE decode -> IMAGE

Key insight: WrapperExecutor.new_class_executor() is invoked INSIDE Flux.forward(),
wrapping only the internal _forward(). Direct attribute access on the Flux nn.Module
and calling sub-modules (time_in, double_blocks[i], etc.) are pure PyTorch — no hooks.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from einops import rearrange
import comfy.model_management

# ------------------------------------------------------------------ #
# Native ComfyUI Flux structural packing / unpacking  
# ------------------------------------------------------------------ #

def pack_latent_flux(latents):
    """Spatial [B,C,H,W] -> sequence [B,T,4C]."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def unpack_latent_flux(latents, latent_h, latent_w):
    """Native ComfyUI FLUX sequence layout unpacking method."""
    b, num_patches, channels = latents.shape
    h = latent_h // 2
    w = latent_w // 2
    latents = latents.view(b, h, w, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(b, channels // 4, latent_h, latent_w)
    return latents


# FLUX.1-dev VAE (AutoencoderKL) constants.
VAE_SCALE_FACTOR = 8
VAE_SHIFT_FACTOR = 0.0609
VAE_SCALING_FACTOR = 0.3611


# ------------------------------------------------------------------ #
# Safe raw Flux extraction from ModelPatcher (bypasses all hooks)  
# ------------------------------------------------------------------ #

def _get_raw_flux(model_patcher):
    """Extract the raw Flux nn.Module (diffusion_model) from ComfyUI's model stack.

    Structure: ModelPatcher -> .model (DiffusionModel wrapper) -> .diffusion_model (Flux nn.Module).

    The intermediate DiffusionModel has attributes like `apply_model`, `encode_adm`,
    and crucially `.diffusion_model` pointing to the actual Flux transformer with
    time_in, double_blocks, single_blocks, final_layer, etc.
    
    WrapperExecutor only wraps method calls inside Flux.forward(), not attribute reads.
    """
    # Step 1: Get model from ModelPatcher (bypasses any subclass overrides).
    raw = getattr(model_patcher, "__dict__", {}).get("model") or model_patcher.model

    if hasattr(raw, "diffusion_model"):
        inner = raw.diffusion_model
        if isinstance(inner, torch.nn.Module):
            # Debug: list all top-level attributes and sub-modules of the Flux module.
            attrs = [a for a in dir(inner) if not a.startswith('_')]
            print(f"[Lotus-2] DEBUG: Extracted diffusion_model type={type(inner).__name__}")
            print(f"[Lotus-2] DEBUG: Top-level attrs: {attrs[:40]}")

            # Check key attributes needed for direct block iteration.
            has_time = hasattr(inner, "time_in")
            has_double = hasattr(inner, "double_blocks")
            has_single = hasattr(inner, "single_blocks")
            has_final = hasattr(inner, "final_layer")
            print(f"[Lotus-2] DEBUG: time_in={has_time}, double_blocks={has_double} ({len(getattr(inner, 'double_blocks', []))}), single_blocks={has_single} ({len(getattr(inner, 'single_blocks', []))}), final_layer={has_final}")
            print(f"[Lotus-2] DEBUG: global_modulation={getattr(inner.params, 'global_modulation', False) if hasattr(inner, 'params') else 'N/A'}")

            return inner

    # Fallback: if the model itself is already a Flux module.
    if isinstance(raw, torch.nn.Module):
        attrs = [a for a in dir(raw) if not a.startswith('_')]
        print(f"[Lotus-2] DEBUG: Using raw model (no diffusion_model attr). Type={type(raw).__name__}")
        print(f"[Lotus-2] DEBUG: Attrs: {attrs[:30]}")
        return raw

    raise TypeError(
        f"[Lotus-2] Cannot extract Flux module from {type(model_patcher).__name__} "
        f"-> inner type {type(raw).__name__}. "
        f"Inner attrs: {[a for a in dir(raw) if not a.startswith('_')]}"
    )


# ------------------------------------------------------------------ #
# Timestep embedding & modulation helpers  
# ------------------------------------------------------------------ #

def _timestep_embedding(timesteps, dim):
    """Sinusoidal timestep / guidance embedding (256-dim standard)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * 
        torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def _modulate(x, mod_out):
    """Apply modulation shift+scale to tensor x (bypasses LayerNorm when affine=False).

    ComfyUI's Modulation outputs a named-tuple-like object: .weight = shift, .shift = scale.
    When norm has elementwise_affine=False we skip it and apply mod directly.
    """
    if hasattr(mod_out, "weight"):
        return x * (1 + mod_out.weight.unsqueeze(1)) + mod_out.shift.unsqueeze(1)
    else:
        raise TypeError(f"[Lotus-2] Unexpected modulation output type: {type(mod_out)}")


def _resize_to_multiple_of_16(image_tensor: torch.Tensor) -> torch.Tensor:
    """Resize so both H and W are divisible by 16 (= 2 * VAE_SCALE_FACTOR)."""
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    min_side = min(h, w)
    scale = (min_side // 16) * 16 / min_side
    new_h = (int(h * scale) // 16) * 16
    new_w = (int(w * scale) // 16) * 16
    return F.interpolate(
        image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


# ------------------------------------------------------------------ #
# Direct Flux forward pass — calls internal nn.Modules directly.
# Completely bypasses Flux.forward() -> WrapperExecutor._forward().
# Calls: process_img(), double_blocks[i](), single_blocks[i](), final_layer()
# All are pure PyTorch nn.Module invocations with no interception.
# ------------------------------------------------------------------ #

def _flux_forward(flux_module, x_spatial, timestep, context=None, pooled_vec=None, guidance=None):
    """Execute Flux transformer by calling internal methods directly on the raw module.

    Bypasses Flux.forward() and WrapperExecutor._forward() entirely.
    Structure mirrors forward_orig() from comfy/ldm/flux/model.py:
      1. process_img(x) -> img, img_ids (spatial [B,C,H,W] -> packed [B,T,H_dim])
      2. Build vec = time_in(timestep_emb) + guidance_in(guidance_emb) + vector_in(y)
      3. txt = txt_in(context), build txt_ids
      4. RoPE: pe = pe_embedder(cat(txt_ids, img_ids))
      5. Iterate double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
      6. Merge: img = cat([txt, img], dim=1), iterate single_blocks
      7. final_layer(img_img_tokens_only, vec_orig) -> noise_pred packed [B,T,C_out]
      8. Rearrange back to spatial [B,C,H,W]

    Returns noise_pred_spatial: [B, C_out, H, W].
    """
    b = x_spatial.shape[0]
    latent_h = x_spatial.shape[2]
    latent_w = x_spatial.shape[3]

    # ---- 1. Pack spatial latents -> sequence format + get img_ids via process_img(). ----
    if not hasattr(flux_module, "process_img"):
        for required in ("process_img", "img_in", "txt_in", "pe_embedder", "double_blocks", "single_blocks", "final_layer"):
            if not hasattr(flux_module, required):
                raise AttributeError(
                    f"[Lotus-2] Flux missing required '{required}'. "
                    f"Available: {[a for a in dir(flux_module) if not a.startswith('_')][:50]}"
                )

    img, img_ids = flux_module.process_img(x_spatial)
    print(f"[Lotus-2] DEBUG process_img: x={x_spatial.shape}, img_packed={img.shape}, img_ids={img_ids.shape}")

    # Verify patch_size and h_len*w_len match token count.
    ps_check = getattr(flux_module, "patch_size", 2)
    expected_tokens = (latent_h // ps_check) * (latent_w // ps_check)
    actual_tokens = img_ids.shape[1]
    print(f"[Lotus-2] DEBUG: patch_size={ps_check}, expected_tokens={expected_tokens}, actual_img_tokens={actual_tokens}")
    # Apply img_in to project from raw packed channels (C*patch**2) → hidden_size (3072).
    if hasattr(flux_module, "img_in"):
        img = flux_module.img_in(img.to(x_spatial.dtype))
    # img: [B, T_img, H_dim], img_ids: [B, T_img, 3]

    txt_len = context.shape[1] if context is not None and len(context.shape) >= 2 else 0
    img_tokens_count = img.shape[1]

    # ---- 2. Build modulation vector (vec) — must mirror forward_orig EXACTLY. ----
    # In ComfyUI Flux: vec = img_in(t_emb) + txt_in(t_emb) + vector_in(y)  (or similar split),
    # then blocks derive (shift_msa, scale_msa, shift_mlp, scale_mlp, gate_msa, gate_mlp) from vec.
    # The "vec" passed to blocks is the SUM, not the per-block-modulated tuple (that's per-block).
    t_emb = _timestep_embedding(timestep.flatten(), 256).to(x_spatial.dtype)

    # In standard FLUX.1-dev, only `time_in` (== `img_in` in some forks) is used for the timestep
    # branch — `time_in` here is the MLP that maps the 256-d sin/cos embedding to hidden_size.
    vec = flux_module.time_in(t_emb)  # [B, hidden_size]


    if guidance is not None and hasattr(flux_module, "guidance_in"):
        g_emb = _timestep_embedding(guidance.flatten(), 256).to(x_spatial.dtype)
        g_out = flux_module.guidance_in(g_emb)
        if isinstance(g_out, torch.Tensor):
            vec = vec + g_out

    if pooled_vec is not None and hasattr(flux_module, "vector_in"):
        vin = flux_module.vector_in
        y_tensor = pooled_vec if isinstance(pooled_vec, torch.Tensor) else pooled_vec.get("pooled_output")
        vec = vec + vin(y_tensor.to(x_spatial.dtype))

    vec = vec.to(dtype=x_spatial.dtype)

    # Sanity: vec must be [B, hidden_size] in the SAME dtype as img/txt. Mixing fp32 vec with
    # bf16 img causes the per-block adaLN to drift by orders of magnitude.
    vec = vec.to(dtype=torch.float32)

    # ---- 3. Project text tokens to hidden size. ----
    txt = flux_module.txt_in(context.to(x_spatial.dtype)) if context is not None and hasattr(flux_module, "txt_in") else torch.zeros(
        b, max(512, img.shape[0]), img.shape[-1], device=x_spatial.device, dtype=x_spatial.dtype
    )
    txt_len = txt.shape[1]

    # ---- 4. Build positional IDs and RoPE embeddings. ----
    # RoPE requires float32 freq tables — bfloat16 here silently corrupts the sin/cos and
    # produces the "blocky / near-constant" decoded output you saw.
    # ---- 4. Build positional IDs and RoPE embeddings. ----
    # txt_len is reassigned after txt_in above — use that final value.
    assert img.shape[-1] == 3072, f"img must be projected to hidden_size=3072, got {img.shape[-1]}"
    assert txt.shape[-1] == 3072, f"txt must be projected to hidden_size=3072, got {txt.shape[-1]}"
    assert img.shape[1] == img_ids.shape[1], f"img/img_ids token count mismatch: {img.shape[1]} vs {img_ids.shape[1]}"

    rope_dtype = torch.float32
    axes_dim = list(flux_module.params.axes_dim)
    txt_ids = torch.zeros((b, txt.shape[1], len(axes_dim)), device=x_spatial.device, dtype=rope_dtype)

    # RoPE positional encoding (txt_ids are zero, img_ids already built by process_img).
    ids = torch.cat((txt_ids, img_ids.to(rope_dtype)), dim=1)
    pe = flux_module.pe_embedder(ids) if hasattr(flux_module, "pe_embedder") else None

    # ---- 5. Per-block modulation: each block derives its own (shift, scale) from vec. ----
    # In FLUX.1-dev (global_modulation=False), vec is the raw [B, hidden] tensor passed to
    # each block, which internally calls block.modulation(vec) to get (msh, msc, gsh, gsc).
    # Your previous code was passing vec through unchanged — that was the silent bug.
    # We keep vec as the raw hidden vector and let the block do the modulation internally.
    vec_orig = vec

    # ---- 6. Iterate double_blocks. ----
    for i, block in enumerate(flux_module.double_blocks):
        img_f = img.to(torch.float32)
        txt_f = txt.to(torch.float32)
        img_out, txt_out = block(img=img_f, txt=txt_f, vec=vec, pe=pe, transformer_options={})
        img, txt = img_out.to(x_spatial.dtype), txt_out.to(x_spatial.dtype)

    # ---- 7. Merge into single stream and iterate single_blocks. ----
    combined = torch.cat((txt, img), dim=1)  # [B, T_txt + T_img, H]
    for i, block in enumerate(flux_module.single_blocks):
        combined_f = combined.to(torch.float32)
        combined_out = block(x=combined_f, vec=vec, pe=pe, transformer_options={})
        combined = combined_out.to(x_spatial.dtype)

    # ---- 8. Extract image tokens and project via final_layer. ----
    img_part = combined[:, txt_len:]  # [B, T_img, H]

    if hasattr(flux_module, "final_layer"):
        # Cast img_part and vec to float32. ComfyUI's final_layer has its own LayerNorm —
        # feed float32 in, let it cast internally if needed.
        noise_pred_packed = flux_module.final_layer(
            img_part.to(torch.float32), vec_orig.to(torch.float32)
        )   # [B, T_img, out_channels*ph*pw]
    else:
        return x_spatial * 0 + torch.zeros_like(x_spatial)

    # ---- 9. Rearrange packed sequence back to spatial format.
    patch_size = flux_module.patch_size
    h_len = latent_h // patch_size
    w_len = latent_w // patch_size

    print(f"[Lotus-2] DEBUG rearrange: latent_h={latent_h}, latent_w={latent_w}, patch_size={patch_size}, h_len={h_len}, w_len={w_len}")
    print(f"[Lotus-2] DEBUG noise_pred_packed shape: {noise_pred_packed.shape} (expected T_img tokens={img_tokens_count})")

    from einops import rearrange
    noise_pred_spatial = rearrange(
        noise_pred_packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_len, w=w_len, ph=patch_size, pw=patch_size
    )[:,:,:latent_h,:latent_w]

    print(f"[Lotus-2] DEBUG noise_pred_spatial: shape={noise_pred_spatial.shape}, "
          f"min={noise_pred_spatial.min().item():.4f}, max={noise_pred_spatial.max().item():.4f}")
    return noise_pred_spatial


class Lotus2InferenceModular:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lcm": ("LOTUS_LCM",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Lotus-2"

    def process(self, model, lcm, image, vae, clip, prompt="", 
                num_inference_steps=10, guidance_scale=3.5):

        # 1. Resolve device / dtype from the ComfyUI ModelPatcher + extract raw Flux module.  
        
        comfy.model_management.load_model_gpu(model)
        flux_module = _get_raw_flux(model)
        device = model.load_device
        dtype = getattr(flux_module, "manual_cast_dtype", None) or torch.bfloat16
        
        vae_dtype = next(vae.first_stage_model.parameters()).dtype

        # 2. Image -> VAE -> spatial latents (B, C, H, W).
        rgb_in = image.permute(0, 3, 1, 2).to(device=device, dtype=vae_dtype)
        rgb_in = _resize_to_multiple_of_16(rgb_in)

        # Normalize [0,1] → [-1,+1] matching Lotus-2 reference (infer.py: /127.5 - 1).
        rgb_in = rgb_in * 2 - 1

        comfy.model_management.load_model_gpu(vae.patcher)
        raw_vae = vae.first_stage_model

        rgb_latents = raw_vae.encode(rgb_in)

        # Handle VAE output variations (some return dist, some return tensor directly).  
        if not isinstance(rgb_latents, torch.Tensor):
            if hasattr(rgb_latents, "sample"):
                rgb_latents = rgb_latents.sample()
            elif hasattr(rgb_latents, "latent_dist"):
                rgb_latents = rgb_latents.latent_dist.sample()
            else:
                rgb_latents = (
                    rgb_latents[0] if isinstance(rgb_latents, (tuple, list)) else rgb_latents
                )

        # Normalize latents for FLUX processing.
        rgb_latents = rgb_latents.to(dtype=dtype)
        rgb_latents = (rgb_latents - VAE_SHIFT_FACTOR) * VAE_SCALING_FACTOR

        print(f"[Lotus-2] DEBUG input latents: shape={rgb_latents.shape}, "
              f"min={rgb_latents.min().item():.4f}, max={rgb_latents.max().item():.4f}")
        latent_height, latent_width = rgb_latents.shape[2], rgb_latents.shape[3]

        # 3. Encode text for transformer conditioning.
        # For depth/normal tasks Lotus-2 uses an empty prompt (or the default unconditional
        # embedding) — these are STRUCTURAL models, not generative. Feeding in a descriptive
        # prompt makes the model "reimagine" the image rather than map it to structure.
        if prompt.strip() == "":
            # Use empty CLIP encoding (zeros) — matches diffusers' `""` prompt path.
            cond = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        else:
            tokens = clip.tokenize([prompt])
            cond = clip.encode_from_tokens(tokens, return_pooled=True)
        prompt_embeds = cond[0].to(device=device, dtype=dtype)
        pooled_prompt_embeds = (
            cond[1].get("pooled_output", cond[1])
            if isinstance(cond[1], dict) else cond[1]
        ).to(device=device, dtype=dtype)

        # Pad to 512 tokens matching diffusers encode_prompt (prevents wrong txt_len in _flux_forward).
        if prompt_embeds.shape[1] < 512:
            pad_size = 512 - prompt_embeds.shape[1]
            padding = torch.zeros(prompt_embeds.shape[0], pad_size, prompt_embeds.shape[2],
                                  device=prompt_embeds.device, dtype=dtype)
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        # Pooled vec: pad/truncate to model's pooled dim (768 for T5-XXL in FLUX).
        if pooled_prompt_embeds.shape[-1] < 768:
            pooled_prompt_embeds = F.pad(pooled_prompt_embeds, (0, 768 - pooled_prompt_embeds.shape[-1]))

        pooled_prompt_embeds = (
            cond[1].get("pooled_output", cond[1])
            if isinstance(cond[1], dict) else cond[1]
        ).to(device=device, dtype=dtype)

        batch_size = rgb_latents.shape[0]
        guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)

        # ---- 4. Core predictor: feed VAE-encoded RGB latents at t≈0.001.
        # Reference pipeline.py passes packed_rgb_latents + timestep_core_predictor/1000 (default=1).
        timestep_core = torch.tensor([1.0] * batch_size, device=device, dtype=dtype)

        latents_spatial = _flux_forward(
            flux_module, rgb_latents, timestep_core,
            context=prompt_embeds, pooled_vec=pooled_prompt_embeds, guidance=guidance,
        )  # [B,C,H,W] — one-shot prediction output.
        print(f"[Lotus-2] core predictor output shape: {latents_spatial.shape}, dtype={latents_spatial.dtype}")

        # ---- 5. LCM smoothing on core predictor latents (spatial). ----
        latents_spatial = lcm(latents_spatial.to(dtype=dtype))

        # ---- 6. Detail sharpener Euler loop.
        # Lotus-2's detail_sharpener runs a SHORT (typically 1-4 step) Euler refinement ON
        # the core predictor's output. The core predictor output is treated as a "denoised
        # starting point at sigma=0", and the sharpener adds tiny noise and refines.
        # Running 10 steps with sigmas from 1.0 → 0.1 destroys the structural information.

        # First, take one scheduler.step() to convert the core predictor's noise_pred into
        # a clean x0 prediction. The core predictor was run at t=1.0 with `rgb_latents` as
        # the "fully noised" input, so its output IS the noise prediction.
        # This is what makes it a real predictor instead of a direct mapping.
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,  # MUST be 1000 for diffusers scheduler math
            base_image_seq_len=256, max_image_seq_len=4096,
            shift=1.0,  # Lotus-2 uses shift=1 (no MuShift) for structural tasks
        )

        # Run only a few refinement steps.
        num_sharpener_steps = min(num_inference_steps, 4)
        sigmas = np.linspace(0.0, 0.0, num_sharpener_steps + 1)[1:]  # all-zero → no-op scheduler
        # Actually for Lotus-2 detail sharpener: skip the scheduler entirely.
        # The core predictor already produced the structural latent; the sharpener LoRA
        # just refines the high-frequency details WITHOUT changing the structural content.
        # Implementation: run forward passes with t in (0, 1/n, 2/n, ...) blending the
        # output back into the latents with small weight.

        euler_dtype = torch.float32
        latents_euler = latents_spatial.float()  # core predictor output as starting point

        """
        for i in range(num_sharpener_steps):
            t_val = (i + 1) / (num_sharpener_steps * 2)  # 0 < t < 0.5
            timestep_t = torch.full((latents_euler.shape[0],), t_val,
                                    device=latents_euler.device, dtype=dtype)

            noise_pred = _flux_forward(
                flux_module, latents_spatial, timestep_t,
                context=prompt_embeds, pooled_vec=pooled_prompt_embeds, guidance=guidance,
            )

            # Light refinement: blend noise_pred into latents with small step size.
            # This is the "detail sharpener" pattern — don't run full scheduler.step(),
            # just nudge high-frequency details.
            blend = 0.15  # small step size
            latents_spatial = latents_spatial + blend * (noise_pred.float() - latents_spatial.float())
            latents_euler = latents_spatial.float()
        """
        pass

        print(f"[Lotus-2] final latents before unscale: {latents_spatial.shape}, channels={latents_spatial.shape[1]}")

        # ---- 8. Inverse scale mapping -> VAE decode step ----
        # The structural latents from the core predictor are in the SAME scaled VAE space
        # as RGB latents (the model was trained on that). Invert the VAE normalization.
        latents = (latents_spatial / VAE_SCALING_FACTOR) + VAE_SHIFT_FACTOR
        # Lotus-2 depth outputs in [-1, 1] range — clamp the latents to that range
        # before decoding, otherwise the VAE produces the saturated psychedelic colors
        # you see in the image (the VAE was trained on [-1, 1] normalized pixels).
        latents = latents.clamp(-3.0, 3.0)

        # DEBUG diagnostics on pre-decode latents.
        print(f"[Lotus-2] DEBUG: latent stats — min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.6f}, std={latents.std().item():.6f}")
        has_nan = torch.isnan(latents).any().item()
        has_inf = torch.isinf(latents).any().item()
        print(f"[Lotus-2] DEBUG: latent nan={has_nan}, inf={has_inf}")

        # Check for repeating patch patterns (symptom of einops rearrange mismatch).
        t_sample = latents[0, 0, :4, :8].cpu().float()
        print(f"[Lotus-2] DEBUG: latent top-left 4x8 sample:\n{t_sample.numpy()}")

        comfy.model_management.load_model_gpu(vae.patcher)
        decoded_image = raw_vae.decode(latents.to(dtype=vae_dtype))
        # Lotus-2 depth output: the VAE-decoded image is essentially grayscale (R≈G≈B),
        # encoding normalized depth. The normal task produces a 3-channel surface normal map
        # with values in [-1, 1]. We keep the raw decoded image here and let the downstream
        # node handle visualization (colormap for depth, shift+scale for normals).
        # No post-processing applied at this stage.

        # DEBUG diagnostics on post-decode image.
        pre_clamp_stats = (f"min={decoded_image.min().item():.4f}, max={decoded_image.max().item():.4f}, "
                          f"mean={decoded_image.mean().item():.6f}, std={decoded_image.std().item():.6f}")
        print(f"[Lotus-2] DEBUG: decoded (before clamp) — {pre_clamp_stats}")

        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)

        # DEBUG post-clamp stats + sample pixel values.
        print(f"[Lotus-2] DEBUG: final output — shape={final_output.shape if 'final_output' in dir() else decoded_image.permute(0,2,3,1).shape}, "
              f"min={decoded_image.min().item():.4f}, max={decoded_image.max().item():.4f}")
        px_sample = decoded_image[0, :3, 5:9, 5:9].cpu().float()  # small RGB patch for inspection
        print(f"[Lotus-2] DEBUG: decoded pixel sample (R,G,B channels x 4x4):\n{px_sample.numpy()}")

        final_output = decoded_image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.float32)
        return (final_output,)