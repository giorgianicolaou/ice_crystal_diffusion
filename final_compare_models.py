# from __future__ import annotations

# import os
# import math
# import json
# import random
# from typing import Dict, List, Tuple, Optional, Any

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm


# # =========================================================
# # GLOBAL SETTINGS
# # =========================================================
# TRAJ_QUANTILES = (0.025, 0.975)
# MEAN_CI_Z = 1.96

# # Mask definition: "zero tail" detection in REAL (abs tolerance)
# ZERO_TAIL_ATOL = float(os.getenv("ZERO_TAIL_ATOL", "1.0"))  # set to 1e-6 if zeros are approximate

# # Evaluation subset cap
# MAX_EVAL_CONDITIONS = os.getenv("MAX_EVAL_CONDITIONS")
# MAX_EVAL_CONDITIONS = int(MAX_EVAL_CONDITIONS) if MAX_EVAL_CONDITIONS else 50

# # Trajectory plot subset cap
# MAX_PLOT_CONDITIONS = os.getenv("MAX_PLOT_CONDITIONS")
# MAX_PLOT_CONDITIONS = int(MAX_PLOT_CONDITIONS) if MAX_PLOT_CONDITIONS else 8

# # Colors
# COLOR_REAL = "black"
# COLOR_MAP_DEFAULT = {
#     "Unconditional": "tab:blue",
#     "Combined": "tab:red",
#     "2D traits": "tab:orange",
#     "VAE only": "tab:green",
# }

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")


# # =========================================================
# # SEEDING
# # =========================================================
# def set_seeds(seed: int = 0) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# set_seeds(0)


# # =========================================================
# # SHAPE HELPERS
# # =========================================================
# def ensure_real_ktd(real: torch.Tensor, d_expected: int) -> torch.Tensor:
#     """
#     Ensure real is (K,T,D). Accepts (K,D,T) and permutes.
#     """
#     if real.ndim != 3:
#         raise ValueError(f"Expected real (K,T,D), got {tuple(real.shape)}")

#     k, a, b = real.shape
#     if b == d_expected:
#         return real.contiguous()  # (K,T,D)
#     if a == d_expected:
#         return real.permute(0, 2, 1).contiguous()  # (K,D,T)->(K,T,D)

#     raise ValueError(f"Cannot infer real layout: shape={tuple(real.shape)} d_expected={d_expected}")


# def ensure_synth_ksdt(synth: torch.Tensor, d_expected: int) -> torch.Tensor:
#     """
#     Ensure synth is (K,S,D,T). Accepts (K,S,T,D) and permutes.
#     """
#     if synth.ndim != 4:
#         raise ValueError(f"Expected synth (K,S,D,T), got {tuple(synth.shape)}")

#     k, s, a, b = synth.shape
#     if a == d_expected:
#         return synth.contiguous()  # (K,S,D,T)
#     if b == d_expected:
#         return synth.permute(0, 1, 3, 2).contiguous()  # (K,S,T,D)->(K,S,D,T)

#     raise ValueError(f"Cannot infer synth layout: shape={tuple(synth.shape)} d_expected={d_expected}")


# # =========================================================
# # NORMALIZATION
# # =========================================================
# def _reshape_mu_sd_for(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     mu = mu.float()
#     sd = sd.float()
#     if x.ndim == 4:  # (K,S,D,T)
#         return mu.reshape(1, 1, -1, 1), sd.reshape(1, 1, -1, 1)
#     if x.ndim == 3:  # (K,T,D)
#         return mu.reshape(1, 1, -1), sd.reshape(1, 1, -1)
#     raise ValueError(f"Expected 3D/4D, got {tuple(x.shape)}")


# def unnormalize(x_norm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
#     mu_b, sd_b = _reshape_mu_sd_for(x_norm, mu, sd)
#     return x_norm * sd_b + mu_b


# def normalize_from_unnorm(x_unnorm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
#     mu_b, sd_b = _reshape_mu_sd_for(x_unnorm, mu, sd)
#     return (x_unnorm - mu_b) / sd_b


# # =========================================================
# # REAL-DERIVED ZERO-TAIL MASK (per condition k, variable d)
# # =========================================================
# def compute_real_zero_tail_mask_ktd(real_unnorm_ktd: torch.Tensor, atol: float = 0.0) -> torch.Tensor:
#     """
#     Return mask_ktd (K,T,D) True where real has a zero tail:
#       - find first t where real is ~0 and stays ~0 to end
#       - mask [t:] for that (k,d)

#     Uses exact zeros if atol==0.0, else near-zero within atol.
#     """
#     if real_unnorm_ktd.ndim != 3:
#         raise ValueError(f"Expected (K,T,D), got {tuple(real_unnorm_ktd.shape)}")

#     K, T, D = real_unnorm_ktd.shape
#     zero = torch.zeros((), device=real_unnorm_ktd.device, dtype=real_unnorm_ktd.dtype)
#     near0 = torch.isclose(real_unnorm_ktd, zero, atol=float(atol), rtol=0.0)  # (K,T,D)

#     near0_i = near0.to(torch.int32)
#     suffix_all = torch.flip(torch.cumprod(torch.flip(near0_i, dims=[1]), dim=1), dims=[1]).to(torch.bool)  # (K,T,D)
#     start_ok = near0 & suffix_all  # (K,T,D)

#     any_true = start_ok.any(dim=1)  # (K,D)
#     start_idx = start_ok.to(torch.float32).argmax(dim=1)  # (K,D), 0 if none
#     start_idx = torch.where(any_true, start_idx, torch.full_like(start_idx, fill_value=T))

#     t_grid = torch.arange(T, device=real_unnorm_ktd.device).view(1, T, 1)  # (1,T,1)
#     return t_grid >= start_idx.view(K, 1, D)  # (K,T,D)


# def apply_mask_nan_ktd(x_ktd: torch.Tensor, mask_ktd: torch.Tensor) -> torch.Tensor:
#     if x_ktd.shape != mask_ktd.shape:
#         raise ValueError(f"Shape mismatch: x={tuple(x_ktd.shape)} mask={tuple(mask_ktd.shape)}")
#     nan = torch.tensor(float("nan"), device=x_ktd.device, dtype=x_ktd.dtype)
#     return torch.where(mask_ktd, nan, x_ktd)


# def apply_mask_nan_ksdt(x_ksdt: torch.Tensor, mask_ktd: torch.Tensor) -> torch.Tensor:
#     """
#     Apply mask_ktd (K,T,D) to synth x_ksdt (K,S,D,T).
#     """
#     if x_ksdt.ndim != 4:
#         raise ValueError(f"Expected synth (K,S,D,T), got {tuple(x_ksdt.shape)}")
#     K, S, D, T = x_ksdt.shape
#     if mask_ktd.shape != (K, T, D):
#         raise ValueError(f"Mask shape mismatch: mask={tuple(mask_ktd.shape)} expected={(K, T, D)}")

#     mask_ksdt = mask_ktd.permute(0, 2, 1).unsqueeze(1).expand(K, S, D, T)  # (K,S,D,T)
#     nan = torch.tensor(float("nan"), device=x_ksdt.device, dtype=x_ksdt.dtype)
#     return torch.where(mask_ksdt, nan, x_ksdt)


# # =========================================================
# # DIAGNOSTIC: Check for near-zero values
# # =========================================================
# def diagnose_zero_behavior(
#     real_unnorm_ktd: torch.Tensor,
#     synth_unnorm_ksdt: torch.Tensor,
#     var_names: List[str],
#     threshold: float = 0.01,
# ) -> None:
#     """
#     Diagnose if synthetic data has small values where real has zeros.
#     """
#     print("\n" + "="*70)
#     print("ZERO-VALUE DIAGNOSTIC")
#     print("="*70)
    
#     K, T, D = real_unnorm_ktd.shape
    
#     for d in range(D):
#         # Count near-zero values in real
#         real_vals = real_unnorm_ktd[:, :, d]
#         real_finite = real_vals[torch.isfinite(real_vals)]
#         real_near_zero = (real_finite.abs() < threshold).sum().item()
#         real_exact_zero = (real_finite == 0.0).sum().item()
        
#         # Count near-zero values in synthetic
#         synth_vals = synth_unnorm_ksdt[:, :, d, :]
#         synth_finite = synth_vals[torch.isfinite(synth_vals)]
#         synth_near_zero = (synth_finite.abs() < threshold).sum().item()
#         synth_exact_zero = (synth_finite == 0.0).sum().item()
        
#         print(f"\n{var_names[d]}:")
#         print(f"  Real:   exact_zeros={real_exact_zero:6d}, "
#               f"near_zeros(<{threshold})={real_near_zero:6d}, "
#               f"total_finite={real_finite.numel():6d}")
#         print(f"  Synth:  exact_zeros={synth_exact_zero:6d}, "
#               f"near_zeros(<{threshold})={synth_near_zero:6d}, "
#               f"total_finite={synth_finite.numel():6d}")
        
#         if real_near_zero > 0 and synth_near_zero > real_near_zero * 2:
#             print(f"  ⚠️  Synthetic has {synth_near_zero/real_near_zero:.1f}x more near-zeros!")


# # =========================================================
# # UNCONDITIONAL ENSEMBLES
# # =========================================================
# def build_unconditional_ensembles_from_total(
#     synth_norm_uncond: torch.Tensor,
#     K_target: int,
#     x: int,
#     d_expected: int,
#     seed: int = 0,
#     allow_wrap: bool = True,
# ) -> torch.Tensor:
#     """
#     From unconditional pool, build (K_target, x, D, T) in normalized space.
#     Accepts synth_norm_uncond as:
#       - (K_u,S_u,D,T) or (K_u,S_u,T,D)
#       - (N,D,T) or (N,T,D)
#     """
#     if synth_norm_uncond.ndim == 4:
#         synth = ensure_synth_ksdt(synth_norm_uncond, d_expected=d_expected)
#         K_u, S_u, D, T = synth.shape
#         pool = synth.reshape(K_u * S_u, D, T)
#     elif synth_norm_uncond.ndim == 3:
#         pool = synth_norm_uncond
#         if pool.shape[1] == d_expected:  # (N,D,T)
#             D, T = pool.shape[1], pool.shape[2]
#         elif pool.shape[2] == d_expected:  # (N,T,D)->(N,D,T)
#             pool = pool.permute(0, 2, 1).contiguous()
#             D, T = pool.shape[1], pool.shape[2]
#         else:
#             raise ValueError(f"Cannot infer unconditional layout from {tuple(pool.shape)}")
#     else:
#         raise ValueError(f"Expected 3D/4D unconditional tensor, got {tuple(synth_norm_uncond.shape)}")

#     N = pool.shape[0]
#     need = K_target * x
#     if N == 0:
#         raise ValueError("Unconditional pool is empty.")

#     g = torch.Generator(device=pool.device)
#     g.manual_seed(seed)
#     perm = torch.randperm(N, generator=g, device=pool.device)

#     if N >= need:
#         chosen = pool[perm[:need]]
#     else:
#         if not allow_wrap:
#             raise ValueError(f"Not enough unconditional samples: have N={N}, need {need}")
#         reps = (need + N - 1) // N
#         chosen = pool[perm.repeat(reps)[:need]]

#     return chosen.reshape(K_target, x, D, T)  # (K_target,S,D,T) with S=x


# # =========================================================
# # LOAD SAMPLES
# # =========================================================
# def load_generated_samples(
#     samples_dir: str,
#     var_names: List[str],
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
#     """
#     Returns:
#       real_unnorm_ref: (K,T,D) UNNORMALIZED
#       synth_by_model_norm: model->(K,S,D,T) NORMALIZED
#       scalers_by_model: model->(mu,sd)
#     """
#     d_expected = len(var_names)
#     print(f"Loading generated samples from: {samples_dir}")

#     name_map = {
#         "combined": "Combined",
#         "vae_only": "VAE only",
#         "2d_traits": "2D traits",
#         "unconditional": "Unconditional",
#     }

#     synth_by_model_norm: Dict[str, torch.Tensor] = {}
#     scalers_by_model: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
#     real_raw: Optional[torch.Tensor] = None
#     mu_ref: Optional[torch.Tensor] = None
#     sd_ref: Optional[torch.Tensor] = None

#     for filename in os.listdir(samples_dir):
#         if not filename.endswith("_synthetic.pt"):
#             continue

#         model_key = filename.replace("_synthetic.pt", "")
#         model_name = name_map.get(model_key, model_key)

#         synth_path = os.path.join(samples_dir, f"{model_key}_synthetic.pt")
#         synth_norm = torch.load(synth_path, map_location="cpu").float()
#         if synth_norm.ndim == 4:
#             synth_norm = ensure_synth_ksdt(synth_norm, d_expected=d_expected)

#         norm_path = os.path.join(samples_dir, f"{model_key}_normalization.pt")
#         norm_data = torch.load(norm_path, map_location="cpu")
#         mu = norm_data["mean"].float()
#         sd = norm_data["std"].float()

#         synth_by_model_norm[model_name] = synth_norm
#         scalers_by_model[model_name] = (mu, sd)

#         if mu_ref is None:
#             mu_ref, sd_ref = mu, sd
#             print(f"Using {model_name} normalization as initial reference")

#         if real_raw is None and model_key != "unconditional":
#             real_path = os.path.join(samples_dir, f"{model_key}_real.pt")
#             if os.path.exists(real_path):
#                 real_raw = torch.load(real_path, map_location="cpu").float()

#     if real_raw is None:
#         raise FileNotFoundError("Could not find real data file (*_real.pt)")

#     real_raw = ensure_real_ktd(real_raw, d_expected=d_expected)

#     assert mu_ref is not None and sd_ref is not None

#     # Heuristic for whether real is normalized
#     real_valid = real_raw[torch.isfinite(real_raw)]
#     real_mean = real_valid.mean().item()
#     real_std = real_valid.std().item()
#     is_normalized = abs(real_mean) < 0.5 and 0.5 < real_std < 1.5

#     if is_normalized:
#         print("Real appears NORMALIZED -> unnormalizing with reference params")
#         real_unnorm_ref = unnormalize(real_raw, mu_ref, sd_ref)
#     else:
#         print("Real appears UNNORMALIZED")
#         real_unnorm_ref = real_raw

#     print(f"Loaded {len(synth_by_model_norm)} models.")
#     print(f"Real shape (K,T,D): {tuple(real_unnorm_ref.shape)}")
#     for n, x in synth_by_model_norm.items():
#         print(f"  {n}: synth shape {tuple(x.shape)}")
#     return real_unnorm_ref, synth_by_model_norm, scalers_by_model


# # =========================================================
# # METRICS
# # =========================================================
# def _mean_ci_over_conditions(x_ktd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     mean_t = np.nanmean(x_ktd, axis=0)
#     std_t = np.nanstd(x_ktd, axis=0)
#     n_t = np.sum(np.isfinite(x_ktd), axis=0).astype(np.float64)
#     se_t = np.where(n_t > 0, std_t / np.sqrt(np.maximum(n_t, 1.0)), np.nan)
#     lo = mean_t - MEAN_CI_Z * se_t
#     hi = mean_t + MEAN_CI_Z * se_t
#     return mean_t, lo, hi


# def _mean_abs_pairwise_diff_1d(x: torch.Tensor) -> torch.Tensor:
#     """
#     Compute E|X - X'| for samples x (1D) in O(n log n), ignoring NaNs.
#     Returns NaN if <2 finite samples.
#     """
#     v = x[torch.isfinite(x)]
#     n = v.numel()
#     if n < 2:
#         return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
#     v, _ = torch.sort(v)
#     idx = torch.arange(n, device=v.device, dtype=v.dtype)
#     coeff = 2.0 * idx - float(n) + 1.0  # 0-indexed: 2*i - n + 1
#     s = torch.sum(coeff * v)
#     return (2.0 / (float(n) * float(n))) * s


# def compute_crps_gpu(synth_ksdt: torch.Tensor, real_ktd: torch.Tensor) -> np.ndarray:
#     """
#     synth_ksdt: (K,S,D,T) normalized, may contain NaNs
#     real_ktd: (K,T,D) normalized, may contain NaNs
#     Returns: (K,T,D)
#     """
#     K, S, D, T = synth_ksdt.shape
#     K2, T2, D2 = real_ktd.shape
#     if (K2, T2, D2) != (K, T, D):
#         raise ValueError(f"Shape mismatch: synth={tuple(synth_ksdt.shape)} real={tuple(real_ktd.shape)}")

#     out = np.full((K, T, D), np.nan, dtype=np.float64)

#     for k in tqdm(range(K), desc="Computing CRPS"):
#         synth_k = synth_ksdt[k].to(DEVICE)          # (S,D,T)
#         real_k = real_ktd[k].to(DEVICE)            # (T,D)

#         # term1: E|X - y|
#         # align real to (1,1,D,T)
#         real_dt = real_k.permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,D,T)
#         diff = torch.abs(synth_k.unsqueeze(0) - real_dt)          # (1,S,D,T)

#         # if real is NaN at (d,t), diff should be NaN
#         real_nan = torch.isnan(real_dt)
#         nan = torch.tensor(float("nan"), device=DEVICE, dtype=diff.dtype)
#         diff = torch.where(real_nan.expand_as(diff), nan, diff)
#         term1 = torch.nanmean(diff, dim=1).squeeze(0)  # (D,T)

#         # term2: E|X - X'| for each (d,t)
#         term2 = torch.empty((D, T), device=DEVICE, dtype=synth_k.dtype)
#         for d in range(D):
#             for t in range(T):
#                 term2[d, t] = _mean_abs_pairwise_diff_1d(synth_k[:, d, t])

#         crps_dt = term1 - 0.5 * term2  # (D,T)
#         out[k] = crps_dt.permute(1, 0).detach().cpu().numpy()  # (T,D)

#     return out


# def compute_variance_over_samples_gpu(synth_ksdt: torch.Tensor, batch_size: int = 50) -> np.ndarray:
#     """
#     synth_ksdt: (K,S,D,T) normalized (masked with NaNs)
#     Returns: (K,T,D)
#     """
#     K, S, D, T = synth_ksdt.shape
#     out = np.zeros((K, T, D), dtype=np.float64)

#     for k0 in range(0, K, batch_size):
#         k1 = min(k0 + batch_size, K)
#         batch = synth_ksdt[k0:k1].to(DEVICE)  # (B,S,D,T)
#         valid = torch.isfinite(batch)
#         n = valid.sum(dim=1).float()  # (B,D,T)

#         batch0 = torch.where(valid, batch, torch.zeros_like(batch))
#         mean = batch0.sum(dim=1) / torch.clamp(n, min=1.0)  # (B,D,T)

#         diff2 = (batch0 - mean.unsqueeze(1)) ** 2
#         diff2 = torch.where(valid, diff2, torch.zeros_like(diff2))
#         var = diff2.sum(dim=1) / torch.clamp(n - 1.0, min=1.0)  # (B,D,T)

#         nan = torch.tensor(float("nan"), device=DEVICE, dtype=var.dtype)
#         var = torch.where(n > 1.0, var, nan)
#         out[k0:k1] = var.permute(0, 2, 1).detach().cpu().numpy()  # (B,T,D)

#     return out


# def _safe_hist_prob(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
#     vals = vals[np.isfinite(vals)]
#     if vals.size == 0:
#         return np.full((len(bins) - 1,), 1.0 / (len(bins) - 1), dtype=np.float64)
#     counts, _ = np.histogram(vals, bins=bins)
#     p = counts.astype(np.float64)
#     s = p.sum()
#     if s <= 0:
#         return np.full_like(p, 1.0 / p.size)
#     return p / s


# def jsd_from_samples(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
#     a = a[np.isfinite(a)]
#     b = b[np.isfinite(b)]
#     if a.size == 0 or b.size == 0:
#         return float("nan")

#     lo = np.nanpercentile(np.concatenate([a, b]), 1)
#     hi = np.nanpercentile(np.concatenate([a, b]), 99)
#     if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
#         return float("nan")

#     edges = np.linspace(lo, hi, bins + 1)
#     p = _safe_hist_prob(a, edges)
#     q = _safe_hist_prob(b, edges)

#     eps = 1e-12
#     p = np.clip(p, eps, 1.0)
#     q = np.clip(q, eps, 1.0)
#     p = p / p.sum()
#     q = q / q.sum()
#     m = 0.5 * (p + q)

#     kl_pm = np.sum(p * np.log(p / m))
#     kl_qm = np.sum(q * np.log(q / m))
#     return float(0.5 * (kl_pm + kl_qm))


# def compute_jsd_per_variable(
#     real_unnorm_ktd: torch.Tensor,     # (K,T,D) unnormalized masked
#     synth_norm_ref_ksdt: torch.Tensor, # (K,S,D,T) normalized masked (reference space)
#     mu_ref: torch.Tensor,
#     sd_ref: torch.Tensor,
#     var_names: List[str],
# ) -> Dict[str, float]:
#     synth_unnorm = unnormalize(synth_norm_ref_ksdt, mu_ref, sd_ref)  # (K,S,D,T)
#     out: Dict[str, float] = {}
#     K, T, D = real_unnorm_ktd.shape
#     for d in range(D):
#         real_vals = real_unnorm_ktd[:, :, d].detach().cpu().numpy().reshape(-1)
#         gen_vals = synth_unnorm[:, :, d, :].detach().cpu().numpy().reshape(-1)
#         out[var_names[d]] = jsd_from_samples(gen_vals, real_vals, bins=200)
#     return out


# # =========================================================
# # PLOTS
# # =========================================================
# def _kde1d(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
#     data = data[np.isfinite(data)]
#     if data.size == 0:
#         return np.zeros_like(grid)
#     std = np.std(data)
#     if std == 0:
#         dens = np.zeros_like(grid)
#         dens[np.argmin(np.abs(grid - data.mean()))] = 1.0
#         return dens
#     n = data.size
#     bw = 1.06 * std * (n ** (-1.0 / 5.0))
#     diffs = (grid[None, :] - data[:, None]) / bw
#     kern = np.exp(-0.5 * diffs**2) / (math.sqrt(2 * math.pi) * bw)
#     return kern.mean(axis=0)


# def plot_kde_per_variable_all_models(
#     real_unnorm_ktd: torch.Tensor,                      # (K,T,D) unnormalized masked
#     synth_by_model_norm_ref_ksdt: Dict[str, torch.Tensor],  # model->(K,S,D,T) ref norm masked
#     var_names: List[str],
#     outdir: str,
#     color_map: Dict[str, str],
#     mu_ref: torch.Tensor,
#     sd_ref: torch.Tensor,
#     num_grid: int = 256,
# ) -> None:
#     os.makedirs(outdir, exist_ok=True)
#     K, T, D = real_unnorm_ktd.shape

#     for d in range(D):
#         real_vals = real_unnorm_ktd[:, :, d].detach().cpu().numpy().reshape(-1)
#         finite_real = real_vals[np.isfinite(real_vals)]
#         if finite_real.size == 0:
#             continue

#         vmin, vmax = np.nanpercentile(finite_real, [1, 99])
#         pooled: Dict[str, np.ndarray] = {}

#         for name, synth_norm_ref in synth_by_model_norm_ref_ksdt.items():
#             synth_unnorm = unnormalize(synth_norm_ref, mu_ref, sd_ref)
#             vals = synth_unnorm[:, :, d, :].detach().cpu().numpy().reshape(-1)
#             pooled[name] = vals
#             finite = vals[np.isfinite(vals)]
#             if finite.size:
#                 lo, hi = np.nanpercentile(finite, [1, 99])
#                 vmin = min(vmin, lo)
#                 vmax = max(vmax, hi)

#         grid = np.linspace(vmin, vmax, num_grid)
#         dens_real = _kde1d(real_vals, grid)

#         plt.figure(figsize=(10, 5))
#         plt.plot(grid, dens_real, label="Real", linewidth=2, color=COLOR_REAL)

#         for name, vals in pooled.items():
#             plt.plot(grid, _kde1d(vals, grid), label=name, color=color_map.get(name, None))

#         plt.title(f"KDE — {var_names[d]}")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"kde_{var_names[d]}.png"))
#         plt.close()


# def plot_time_series_mean_ci_per_variable(
#     series_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
#     var_names: List[str],
#     outdir: str,
#     title_prefix: str,
#     ylabel: str,
#     color_map: Dict[str, str],
#     fname_prefix: str,
# ) -> None:
#     os.makedirs(outdir, exist_ok=True)
#     example = next(iter(series_by_model.values()))
#     _, T, D = example.shape
#     x = np.arange(T)

#     for d in range(D):
#         plt.figure(figsize=(12, 6))
#         for name, arr in series_by_model.items():
#             x_kt = arr[:, :, d]
#             mean_t, lo, hi = _mean_ci_over_conditions(x_kt)
#             plt.plot(x, mean_t, linewidth=2, label=name, color=color_map.get(name, None))
#             plt.fill_between(x, lo, hi, alpha=0.20, color=color_map.get(name, None))
#         plt.xlabel("Time Step")
#         plt.ylabel(ylabel)
#         plt.title(f"{title_prefix} — {var_names[d]} (mean ± 95% CI)")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{fname_prefix}_{var_names[d]}.png"))
#         plt.close()


# def plot_crps_over_time_no_ci(
#     crps_norm_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
#     var_names: List[str],
#     outdir: str,
#     color_map: Dict[str, str],
# ) -> None:
#     """
#     Plot CRPS over time (mean across conditions) without confidence intervals.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     example = next(iter(crps_norm_by_model.values()))
#     _, T, D = example.shape
#     x = np.arange(T)
    
#     for d in range(D):
#         plt.figure(figsize=(12, 6))
        
#         for name, crps_ktd in crps_norm_by_model.items():
#             crps_td = crps_ktd[:, :, d]  # (K, T)
#             mean_t = np.nanmean(crps_td, axis=0)  # (T,)
            
#             plt.plot(x, mean_t, linewidth=2.5, label=name, 
#                     color=color_map.get(name, None))
        
#         plt.xlabel("Time Step", fontsize=12)
#         plt.ylabel("CRPS (normalized)", fontsize=12)
#         plt.title(f"CRPS Over Time — {var_names[d]}", fontsize=14, fontweight='bold')
#         plt.legend(fontsize=11)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"crps_time_no_ci_{var_names[d]}.png"), dpi=150)
#         plt.close()


# def plot_crps_comparison_barplot(
#     crps_norm_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
#     var_names: List[str],
#     outdir: str,
#     conditional_key: str = "Combined",
#     unconditional_key: str = "Unconditional",
# ) -> None:
#     """
#     Create bar plot comparing average CRPS between conditional and unconditional.
#     """
#     os.makedirs(outdir, exist_ok=True)
    
#     if conditional_key not in crps_norm_by_model or unconditional_key not in crps_norm_by_model:
#         print(f"⚠️  Cannot create comparison: need both {conditional_key} and {unconditional_key}")
#         return
    
#     crps_cond = crps_norm_by_model[conditional_key]
#     crps_uncond = crps_norm_by_model[unconditional_key]
    
#     # === PLOT 1: Overall average ===
#     overall_cond = np.nanmean(crps_cond)
#     overall_uncond = np.nanmean(crps_uncond)
    
#     plt.figure(figsize=(8, 6))
#     x = np.array([0, 1])
#     heights = [overall_cond, overall_uncond]
#     colors_list = [COLOR_MAP_DEFAULT[conditional_key], COLOR_MAP_DEFAULT[unconditional_key]]
    
#     bars = plt.bar(x, heights, width=0.6, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
#     for bar, height in zip(bars, heights):
#         plt.text(bar.get_x() + bar.get_width()/2, height + 0.005,
#                 f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
#     plt.xticks(x, [conditional_key, unconditional_key], fontsize=12)
#     plt.ylabel('Average CRPS (normalized)', fontsize=12)
#     plt.title('Overall CRPS Comparison', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, 'crps_comparison_overall.png'), dpi=150)
#     plt.close()
    
#     # === PLOT 2: Per-variable comparison ===
#     K, T, D = crps_cond.shape
#     per_var_cond = []
#     per_var_uncond = []
    
#     for d in range(D):
#         per_var_cond.append(np.nanmean(crps_cond[:, :, d]))
#         per_var_uncond.append(np.nanmean(crps_uncond[:, :, d]))
    
#     x = np.arange(D)
#     width = 0.35
    
#     fig, ax = plt.subplots(figsize=(14, 6))
    
#     bars1 = ax.bar(x - width/2, per_var_cond, width, label=conditional_key,
#                    color=COLOR_MAP_DEFAULT[conditional_key], alpha=0.8, edgecolor='black')
#     bars2 = ax.bar(x + width/2, per_var_uncond, width, label=unconditional_key,
#                    color=COLOR_MAP_DEFAULT[unconditional_key], alpha=0.8, edgecolor='black')
    
#     ax.set_xlabel('Variable', fontsize=12)
#     ax.set_ylabel('Average CRPS (normalized)', fontsize=12)
#     ax.set_title('CRPS Comparison by Variable', fontsize=14, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=10)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, 'crps_comparison_per_variable.png'), dpi=150)
#     plt.close()
    
#     # === PLOT 3: Improvement percentage ===
#     improvement = []
#     for d in range(D):
#         cond_val = per_var_cond[d]
#         uncond_val = per_var_uncond[d]
#         if not np.isnan(uncond_val) and uncond_val != 0:
#             pct = ((uncond_val - cond_val) / uncond_val) * 100
#             improvement.append(pct)
#         else:
#             improvement.append(0)
    
#     fig, ax = plt.subplots(figsize=(14, 6))
#     colors = ['green' if imp > 0 else 'red' for imp in improvement]
#     bars = ax.bar(x, improvement, color=colors, alpha=0.7, edgecolor='black')
    
#     ax.axhline(0, color='black', linewidth=1, linestyle='-')
#     ax.set_xlabel('Variable', fontsize=12)
#     ax.set_ylabel('Improvement (%)', fontsize=12)
#     ax.set_title('CRPS Improvement: Conditional vs Unconditional\n(Positive = Conditional Better)',
#                 fontsize=14, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=10)
#     ax.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, 'crps_improvement_percentage.png'), dpi=150)
#     plt.close()
    
#     print(f"\n{'='*70}")
#     print("CRPS COMPARISON SUMMARY")
#     print(f"{'='*70}")
#     print(f"Overall Average:")
#     print(f"  {conditional_key:20s}: {overall_cond:.6f}")
#     print(f"  {unconditional_key:20s}: {overall_uncond:.6f}")
#     print(f"  Improvement: {((overall_uncond - overall_cond)/overall_uncond*100):.2f}%")
#     print(f"\nPer-Variable Improvement:")
#     for d, var_name in enumerate(var_names):
#         print(f"  {var_name:12s}: {improvement[d]:+6.2f}%")


# def plot_overall_boxplot(metric_by_model: Dict[str, np.ndarray], outpath: str, title: str, ylabel: str) -> None:
#     model_names = list(metric_by_model.keys())
#     data = []
#     for name in model_names:
#         vals = metric_by_model[name]
#         vals = vals[np.isfinite(vals)]
#         data.append(vals)

#     plt.figure(figsize=(12, 6))
#     plt.boxplot(data, labels=model_names, showfliers=False)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.savefig(outpath)
#     plt.close()


# def plot_per_variable_boxplots(
#     metric_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
#     var_names: List[str],
#     outdir: str,
#     title_prefix: str,
#     ylabel: str,
#     fname_prefix: str,
# ) -> None:
#     os.makedirs(outdir, exist_ok=True)
#     example = next(iter(metric_by_model.values()))
#     _, _, D = example.shape

#     for name, arr in metric_by_model.items():
#         fig = plt.figure(figsize=(max(10, D * 0.7), 6))
#         data: List[np.ndarray] = []
#         for d in range(D):
#             vals = arr[:, :, d].ravel()
#             vals = vals[np.isfinite(vals)]
#             data.append(vals)

#         plt.boxplot(data, showfliers=False)
#         plt.xticks(np.arange(1, D + 1), var_names, rotation=30, ha="right")
#         plt.ylabel(ylabel)
#         plt.title(f"{title_prefix} — {name}")
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{fname_prefix}_{name}.png"))
#         plt.close(fig)


# def plot_variance_over_time(
#     var_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
#     var_names: List[str],
#     outdir: str,
#     color_map: Dict[str, str],
#     fname_prefix: str,
#     title_prefix: str,
# ) -> None:
#     os.makedirs(outdir, exist_ok=True)
#     example = next(iter(var_by_model.values()))
#     _, T, D = example.shape
#     x = np.arange(T)

#     for d in range(D):
#         plt.figure(figsize=(12, 6))
#         for name, arr in var_by_model.items():
#             x_kt = arr[:, :, d]
#             mean_t, lo, hi = _mean_ci_over_conditions(x_kt)
#             plt.plot(x, mean_t, linewidth=2, label=name, color=color_map.get(name, None))
#             plt.fill_between(x, lo, hi, alpha=0.20, color=color_map.get(name, None))
#         plt.title(f"{title_prefix} — {var_names[d]} (mean ± 95% CI)")
#         plt.xlabel("Time Step")
#         plt.ylabel("Variance")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{fname_prefix}_{var_names[d]}.png"))
#         plt.close()


# def _select_condition_indices(K: int) -> List[int]:
#     if K <= 0:
#         return []
#     m = min(MAX_PLOT_CONDITIONS, K)
#     idxs = np.linspace(0, K - 1, m).round().astype(int).tolist()
#     return sorted(set(idxs))


# def select_best_crps_conditions(
#     crps_norm_ktd: np.ndarray,  # (K,T,D)
#     n_select: int = None,
# ) -> List[int]:
#     """
#     Select conditions with LOWEST average CRPS (best performing).
#     """
#     if n_select is None:
#         n_select = MAX_PLOT_CONDITIONS
    
#     K, T, D = crps_norm_ktd.shape
    
#     # Average CRPS over time and variables for each condition
#     crps_per_condition = np.nanmean(crps_norm_ktd, axis=(1, 2))  # (K,)
    
#     # Get indices sorted by CRPS (ascending = best first)
#     sorted_indices = np.argsort(crps_per_condition)
    
#     # Select top n_select
#     n_select = min(n_select, K)
#     best_indices = sorted_indices[:n_select].tolist()
    
#     print(f"\n{'='*70}")
#     print(f"SELECTED BEST {n_select} CONDITIONS BY CRPS")
#     print(f"{'='*70}")
#     for rank, k in enumerate(best_indices, 1):
#         crps_val = crps_per_condition[k]
#         print(f"  Rank {rank:2d}: Condition {k:4d} with CRPS = {crps_val:.6f}")
    
#     return best_indices


# def plot_trajectories_ci_cond_vs_uncond_direct(
#     real_unnorm_ktd: torch.Tensor,
#     synth_cond_unnorm_ksdt: torch.Tensor,
#     synth_uncond_unnorm_ksdt: torch.Tensor,
#     var_names: List[str],
#     outdir: str,
#     crps_norm_ktd: Optional[np.ndarray] = None,
#     n_plot: Optional[int] = None,
# ) -> None:
#     """
#     Plot trajectories with option to select best CRPS conditions.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     K, T, D = real_unnorm_ktd.shape
    
#     if n_plot is None:
#         n_plot = MAX_PLOT_CONDITIONS
    
#     # Select conditions
#     if crps_norm_ktd is not None:
#         print("Selecting best CRPS conditions for trajectory plots...")
#         cond_idxs = select_best_crps_conditions(crps_norm_ktd, n_select=n_plot)
#     else:
#         print("Using default condition selection (uniform spacing)...")
#         cond_idxs = _select_condition_indices(K)
    
#     x = np.arange(T)
    
#     for k in tqdm(cond_idxs, desc="Trajectory plots (cond vs uncond)"):
#         for d in range(D):
#             plt.figure(figsize=(11, 6))
#             real = real_unnorm_ktd[k, :, d].detach().cpu().numpy()
#             plt.plot(x, real, linewidth=2, color=COLOR_REAL, label="Ground truth")
            
#             s_c = synth_cond_unnorm_ksdt[k, :, d, :].detach().cpu().numpy()
#             qlo_c = np.nanquantile(s_c, TRAJ_QUANTILES[0], axis=0)
#             qhi_c = np.nanquantile(s_c, TRAJ_QUANTILES[1], axis=0)
#             mean_c = np.nanmean(s_c, axis=0)
#             plt.fill_between(x, qlo_c, qhi_c, alpha=0.25, 
#                            color=COLOR_MAP_DEFAULT["Combined"], 
#                            label="Conditional 95% interval")
#             plt.plot(x, mean_c, linewidth=2, 
#                     color=COLOR_MAP_DEFAULT["Combined"], 
#                     label="Conditional mean")
            
#             s_u = synth_uncond_unnorm_ksdt[k, :, d, :].detach().cpu().numpy()
#             qlo_u = np.nanquantile(s_u, TRAJ_QUANTILES[0], axis=0)
#             qhi_u = np.nanquantile(s_u, TRAJ_QUANTILES[1], axis=0)
#             mean_u = np.nanmean(s_u, axis=0)
#             plt.fill_between(x, qlo_u, qhi_u, alpha=0.25, 
#                            color=COLOR_MAP_DEFAULT["Unconditional"], 
#                            label="Unconditional 95% interval")
#             plt.plot(x, mean_u, linewidth=2, 
#                     color=COLOR_MAP_DEFAULT["Unconditional"], 
#                     label="Unconditional mean")
            
#             # Add CRPS value in title if available
#             if crps_norm_ktd is not None:
#                 crps_val = np.nanmean(crps_norm_ktd[k, :, d])
#                 title = f"Condition Sample — {var_names[d]}"
#             else:
#                 title = f"Condition Sample — {var_names[d]}"
            
#             plt.title(title, fontsize=12, fontweight='bold')
#             plt.xlabel("Time")
#             plt.ylabel("Value")
#             plt.legend(ncol=2, fontsize=10)
#             plt.tight_layout()
#             plt.savefig(os.path.join(outdir, f"traj_ci_cond{k}_{var_names[d]}.png"))
#             plt.close()


# # =========================================================
# # METRICS EXPORT
# # =========================================================
# def _summarize_metric(arr_ktd: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
#     arr_flat = arr_ktd[np.isfinite(arr_ktd)]
#     _, _, D = arr_ktd.shape

#     per_var_mean: Dict[str, float] = {}
#     per_var_var: Dict[str, float] = {}
#     for d in range(D):
#         vals = arr_ktd[:, :, d].ravel()
#         vals = vals[np.isfinite(vals)]
#         per_var_mean[var_names[d]] = float(np.mean(vals)) if vals.size else float("nan")
#         per_var_var[var_names[d]] = float(np.var(vals)) if vals.size else float("nan")

#     return {
#         "global": {
#             "mean": float(np.mean(arr_flat)) if arr_flat.size else float("nan"),
#             "variance": float(np.var(arr_flat)) if arr_flat.size else float("nan"),
#         },
#         "per_variable": {"mean": per_var_mean, "variance": per_var_var},
#     }


# def export_metrics_json(
#     outpath: str,
#     var_names: List[str],
#     crps_norm_by_model: Dict[str, np.ndarray],
#     crps_unnorm_by_model: Dict[str, np.ndarray],
#     var_norm_by_model: Dict[str, np.ndarray],
#     var_unnorm_by_model: Dict[str, np.ndarray],
#     jsd_by_model: Dict[str, Dict[str, float]],
# ) -> None:
#     stats: Dict[str, Any] = {
#         "crps": {
#             "normalized": {m: _summarize_metric(a, var_names) for m, a in crps_norm_by_model.items()},
#             "unnormalized": {m: _summarize_metric(a, var_names) for m, a in crps_unnorm_by_model.items()},
#         },
#         "variance": {
#             "normalized": {m: _summarize_metric(a, var_names) for m, a in var_norm_by_model.items()},
#             "unnormalized": {m: _summarize_metric(a, var_names) for m, a in var_unnorm_by_model.items()},
#         },
#         "jsd": jsd_by_model,
#     }
#     with open(outpath, "w") as f:
#         json.dump(stats, f, indent=4)


# # =========================================================
# # ANALYSIS SUITE
# # =========================================================
# def run_analysis_suite(
#     suite_name: str,
#     out_root: str,
#     var_names: List[str],
#     real_unnorm_ref: torch.Tensor,
#     synth_by_model_norm: Dict[str, torch.Tensor],
#     scalers_by_model: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#     color_map: Dict[str, str],
#     make_cond_vs_uncond_trajectories: bool = False,
# ) -> None:
#     os.makedirs(out_root, exist_ok=True)
#     d_expected = len(var_names)

#     mu_ref, sd_ref = scalers_by_model.get("Combined", next(iter(scalers_by_model.values())))

#     real_unnorm_ref = ensure_real_ktd(real_unnorm_ref, d_expected=d_expected)
#     synth_std: Dict[str, torch.Tensor] = {n: ensure_synth_ksdt(x, d_expected=d_expected) for n, x in synth_by_model_norm.items()}

#     K_min = real_unnorm_ref.shape[0]
#     for name, x in synth_std.items():
#         if name != "Unconditional":
#             K_min = min(K_min, x.shape[0])

#     K_eval = min(K_min, MAX_EVAL_CONDITIONS) if MAX_EVAL_CONDITIONS else K_min

#     print(f"\n{'='*70}")
#     print(f"Running suite: {suite_name}")
#     print(f"K_min={K_min}, K_eval={K_eval} (MAX_EVAL_CONDITIONS={MAX_EVAL_CONDITIONS})")
#     print(f"ZERO_TAIL_ATOL={ZERO_TAIL_ATOL}  |  MAX_PLOT_CONDITIONS={MAX_PLOT_CONDITIONS}")
#     print(f"{'='*70}\n")

#     real_unnorm_eval = real_unnorm_ref[:K_eval].clone()

#     mask_ktd = compute_real_zero_tail_mask_ktd(real_unnorm_eval, atol=ZERO_TAIL_ATOL)
#     real_unnorm_eval_masked = apply_mask_nan_ktd(real_unnorm_eval, mask_ktd)
#     real_norm_ref_masked = normalize_from_unnorm(real_unnorm_eval_masked, mu_ref, sd_ref)

#     synth_by_model_unnorm_masked: Dict[str, torch.Tensor] = {}
#     synth_by_model_norm_ref_masked: Dict[str, torch.Tensor] = {}

#     for name, synth_norm in synth_std.items():
#         mu, sd = scalers_by_model[name]

#         if name == "Unconditional":
#             S_model = synth_norm.shape[1]
#             synth_norm_trimmed = build_unconditional_ensembles_from_total(
#                 synth_norm_uncond=synth_norm,
#                 K_target=K_eval,
#                 x=S_model,
#                 d_expected=d_expected,
#                 seed=0,
#                 allow_wrap=True,
#             )
#         else:
#             synth_norm_trimmed = synth_norm[:K_eval]

#         synth_unnorm = unnormalize(synth_norm_trimmed, mu, sd)
#         synth_unnorm_masked = apply_mask_nan_ksdt(synth_unnorm, mask_ktd)
#         synth_norm_ref_masked = normalize_from_unnorm(synth_unnorm_masked, mu_ref, sd_ref)

#         synth_by_model_unnorm_masked[name] = synth_unnorm_masked
#         synth_by_model_norm_ref_masked[name] = synth_norm_ref_masked

#     # Run diagnostics
#     print("\n" + "="*70)
#     print("RUNNING ZERO-VALUE DIAGNOSTICS")
#     print("="*70)
#     for name, synth_unnorm_masked in synth_by_model_unnorm_masked.items():
#         print(f"\n{name}:")
#         diagnose_zero_behavior(
#             real_unnorm_eval_masked,
#             synth_unnorm_masked,
#             var_names,
#             threshold=0.01,
#         )

#     # Compute metrics
#     crps_norm_by_model: Dict[str, np.ndarray] = {}
#     crps_unnorm_by_model: Dict[str, np.ndarray] = {}
#     var_norm_by_model: Dict[str, np.ndarray] = {}
#     var_unnorm_by_model: Dict[str, np.ndarray] = {}
#     jsd_by_model: Dict[str, Dict[str, float]] = {}

#     sd_ref_np = sd_ref.view(-1).detach().cpu().numpy().reshape(1, 1, -1)

#     for name, synth_norm_ref in synth_by_model_norm_ref_masked.items():
#         print(f"Computing metrics for {name}...")

#         crps_norm = compute_crps_gpu(synth_norm_ref, real_norm_ref_masked)
#         crps_norm_by_model[name] = crps_norm
#         crps_unnorm_by_model[name] = crps_norm * sd_ref_np

#         v_norm = compute_variance_over_samples_gpu(synth_norm_ref)
#         var_norm_by_model[name] = v_norm
#         var_unnorm_by_model[name] = v_norm * (sd_ref_np ** 2)

#         jsd_by_model[name] = compute_jsd_per_variable(
#             real_unnorm_ktd=real_unnorm_eval_masked,
#             synth_norm_ref_ksdt=synth_norm_ref,
#             mu_ref=mu_ref,
#             sd_ref=sd_ref,
#             var_names=var_names,
#         )

#         print(f"  CRPS(norm)  mean={np.nanmean(crps_norm):.6f}")
#         print(f"  CRPS(unn)   mean={np.nanmean(crps_unnorm_by_model[name]):.6f}")
#         print()

#     # Export metrics
#     metrics_path = os.path.join(out_root, "metrics_summary.json")
#     export_metrics_json(
#         metrics_path,
#         var_names,
#         crps_norm_by_model,
#         crps_unnorm_by_model,
#         var_norm_by_model,
#         var_unnorm_by_model,
#         jsd_by_model,
#     )

#     # PLOTS
#     crps_dir = os.path.join(out_root, "crps")
#     os.makedirs(crps_dir, exist_ok=True)

#     plot_per_variable_boxplots(
#         crps_unnorm_by_model,
#         var_names,
#         outdir=crps_dir,
#         title_prefix="CRPS (unnormalized) per variable",
#         ylabel="CRPS",
#         fname_prefix="crps_boxplots_unnorm",
#     )

#     plot_overall_boxplot(
#         crps_unnorm_by_model,
#         outpath=os.path.join(crps_dir, "crps_overall_boxplot_unnorm.png"),
#         title=f"{suite_name}: CRPS (unnormalized)",
#         ylabel="CRPS",
#     )

#     plot_time_series_mean_ci_per_variable(
#         crps_norm_by_model,
#         var_names,
#         outdir=crps_dir,
#         title_prefix="CRPS (normalized) over time",
#         ylabel="CRPS (normalized)",
#         color_map=color_map,
#         fname_prefix="crps_time_mean_ci_norm",
#     )

#     # NEW: CRPS over time without CI
#     print("Generating CRPS over time (no CI)...")
#     plot_crps_over_time_no_ci(
#         crps_norm_by_model,
#         var_names,
#         outdir=crps_dir,
#         color_map=color_map,
#     )

#     # NEW: CRPS comparison bar plots
#     if "Combined" in crps_norm_by_model and "Unconditional" in crps_norm_by_model:
#         print("Generating CRPS comparison bar plots...")
#         plot_crps_comparison_barplot(
#             crps_norm_by_model,
#             var_names,
#             outdir=crps_dir,
#             conditional_key="Combined",
#             unconditional_key="Unconditional",
#         )

#     var_dir = os.path.join(out_root, "variance")
#     os.makedirs(var_dir, exist_ok=True)

#     plot_variance_over_time(
#         var_unnorm_by_model,
#         var_names,
#         outdir=var_dir,
#         color_map=color_map,
#         fname_prefix="variance_time_mean_ci_unnorm",
#         title_prefix="Variance (unnormalized) over time",
#     )

#     kde_dir = os.path.join(out_root, "kde")
#     plot_kde_per_variable_all_models(
#         real_unnorm_eval_masked,
#         synth_by_model_norm_ref_masked,
#         var_names,
#         kde_dir,
#         color_map=color_map,
#         mu_ref=mu_ref,
#         sd_ref=sd_ref,
#     )

#     if make_cond_vs_uncond_trajectories:
#         if "Combined" in synth_by_model_unnorm_masked and "Unconditional" in synth_by_model_unnorm_masked:
#             traj_dir = os.path.join(out_root, "trajectories_cond_vs_uncond")
            
#             # Get CRPS for condition selection
#             crps_cond = crps_norm_by_model.get("Combined", None)
            
#             plot_trajectories_ci_cond_vs_uncond_direct(
#                 real_unnorm_eval_masked,
#                 synth_by_model_unnorm_masked["Combined"],
#                 synth_by_model_unnorm_masked["Unconditional"],
#                 var_names,
#                 traj_dir,
#                 crps_norm_ktd=crps_cond,
#                 n_plot=MAX_PLOT_CONDITIONS,
#             )
#         else:
#             print("Skipping trajectories: need both Combined and Unconditional.")

#     print(f"\n✓ Suite '{suite_name}' complete. Output: {out_root}\n")


# # =========================================================
# # MAIN
# # =========================================================
# if __name__ == "__main__":
#     var_names = [
#         "WRF_TEMP", "WRF_PRES", "WRF_RELH", "WRF_PHI", "WRF_PHIS",
#         "WRF_QICE", "WRF_QSNOW", "WRF_QVAPOR", "WRF_QCLOUD", "WRF_QRAIN",
#     ]
#     MAX_EVAL_CONDITIONS = 2134
#     samples_dir = "./synthetic_samples"
#     out_base = "analysis_results_simple"
    
#     real_unnorm_ref, synth_by_model_norm, scalers_by_model = load_generated_samples(samples_dir, var_names)

#     run_analysis_suite(
#         suite_name="All models",
#         out_root=os.path.join(out_base, "all_4_models"),
#         var_names=var_names,
#         real_unnorm_ref=real_unnorm_ref,
#         synth_by_model_norm=synth_by_model_norm,
#         scalers_by_model=scalers_by_model,
#         color_map=COLOR_MAP_DEFAULT,
#         make_cond_vs_uncond_trajectories=False,
#     )

#     if "Combined" in synth_by_model_norm and "Unconditional" in synth_by_model_norm:
#         suite_models = {
#             "Unconditional": synth_by_model_norm["Unconditional"],
#             "Combined": synth_by_model_norm["Combined"],
#         }
#         suite_scalers = {
#             "Unconditional": scalers_by_model["Unconditional"],
#             "Combined": scalers_by_model["Combined"],
#         }
#         suite_colors = {
#             "Unconditional": COLOR_MAP_DEFAULT["Unconditional"],
#             "Combined": COLOR_MAP_DEFAULT["Combined"],
#         }

#         run_analysis_suite(
#             suite_name="Conditional vs Unconditional",
#             out_root=os.path.join(out_base, "cond_vs_uncond_only"),
#             var_names=var_names,
#             real_unnorm_ref=real_unnorm_ref,
#             synth_by_model_norm=suite_models,
#             scalers_by_model=suite_scalers,
#             color_map=suite_colors,
#             make_cond_vs_uncond_trajectories=True,
#         )

#     print("\n" + "="*70)
#     print("ALL ANALYSIS COMPLETE!")
#     print("="*70)

from __future__ import annotations

import os
import math
import json
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# GLOBAL SETTINGS
# =========================================================
TRAJ_QUANTILES = (0.025, 0.975)
MEAN_CI_Z = 1.96

# Mask definition: "zero tail" detection in REAL (abs tolerance)
ZERO_TAIL_ATOL = float(os.getenv("ZERO_TAIL_ATOL", "1.0"))  # set to 1e-6 if zeros are approximate

# NEW: Synthetic "near-zero tail" detection threshold (abs tol)
SYNTH_ZERO_TAIL_ATOL = float(os.getenv("SYNTH_ZERO_TAIL_ATOL", "1.00"))

# Evaluation subset cap
MAX_EVAL_CONDITIONS = os.getenv("MAX_EVAL_CONDITIONS")
MAX_EVAL_CONDITIONS = int(MAX_EVAL_CONDITIONS) if MAX_EVAL_CONDITIONS else 50

# Trajectory plot subset cap
MAX_PLOT_CONDITIONS = os.getenv("MAX_PLOT_CONDITIONS")
MAX_PLOT_CONDITIONS = int(MAX_PLOT_CONDITIONS) if MAX_PLOT_CONDITIONS else 8

# Colors
COLOR_REAL = "black"
COLOR_MAP_DEFAULT = {
    "Unconditional": "tab:blue",
    "Combined": "tab:red",
    "2D traits": "tab:orange",
    "VAE only": "tab:green",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================================================
# SEEDING
# =========================================================
def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(0)


# =========================================================
# SHAPE HELPERS
# =========================================================
def ensure_real_ktd(real: torch.Tensor, d_expected: int) -> torch.Tensor:
    """
    Ensure real is (K,T,D). Accepts (K,D,T) and permutes.
    """
    if real.ndim != 3:
        raise ValueError(f"Expected real (K,T,D), got {tuple(real.shape)}")

    k, a, b = real.shape
    if b == d_expected:
        return real.contiguous()  # (K,T,D)
    if a == d_expected:
        return real.permute(0, 2, 1).contiguous()  # (K,D,T)->(K,T,D)

    raise ValueError(f"Cannot infer real layout: shape={tuple(real.shape)} d_expected={d_expected}")


def ensure_synth_ksdt(synth: torch.Tensor, d_expected: int) -> torch.Tensor:
    """
    Ensure synth is (K,S,D,T). Accepts (K,S,T,D) and permutes.
    """
    if synth.ndim != 4:
        raise ValueError(f"Expected synth (K,S,D,T), got {tuple(synth.shape)}")

    k, s, a, b = synth.shape
    if a == d_expected:
        return synth.contiguous()  # (K,S,D,T)
    if b == d_expected:
        return synth.permute(0, 1, 3, 2).contiguous()  # (K,S,T,D)->(K,S,D,T)

    raise ValueError(f"Cannot infer synth layout: shape={tuple(synth.shape)} d_expected={d_expected}")


# =========================================================
# NORMALIZATION
# =========================================================
def _reshape_mu_sd_for(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = mu.float()
    sd = sd.float()
    if x.ndim == 4:  # (K,S,D,T)
        return mu.reshape(1, 1, -1, 1), sd.reshape(1, 1, -1, 1)
    if x.ndim == 3:  # (K,T,D)
        return mu.reshape(1, 1, -1), sd.reshape(1, 1, -1)
    raise ValueError(f"Expected 3D/4D, got {tuple(x.shape)}")


def unnormalize(x_norm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    mu_b, sd_b = _reshape_mu_sd_for(x_norm, mu, sd)
    return x_norm * sd_b + mu_b


def normalize_from_unnorm(x_unnorm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    mu_b, sd_b = _reshape_mu_sd_for(x_unnorm, mu, sd)
    return (x_unnorm - mu_b) / sd_b


# =========================================================
# REAL-DERIVED ZERO-TAIL MASK (per condition k, variable d)
# =========================================================
def compute_real_zero_tail_mask_ktd(real_unnorm_ktd: torch.Tensor, atol: float = 0.0) -> torch.Tensor:
    """
    Return mask_ktd (K,T,D) True where real has a zero tail:
      - find first t where real is ~0 and stays ~0 to end
      - mask [t:] for that (k,d)

    Uses exact zeros if atol==0.0, else near-zero within atol.
    """
    if real_unnorm_ktd.ndim != 3:
        raise ValueError(f"Expected (K,T,D), got {tuple(real_unnorm_ktd.shape)}")

    K, T, D = real_unnorm_ktd.shape
    zero = torch.zeros((), device=real_unnorm_ktd.device, dtype=real_unnorm_ktd.dtype)
    finite = torch.isfinite(real_unnorm_ktd)
    near0 = finite & torch.isclose(real_unnorm_ktd, zero, atol=float(atol), rtol=0.0)  # (K,T,D)

    near0_i = near0.to(torch.int32)
    suffix_all = torch.flip(torch.cumprod(torch.flip(near0_i, dims=[1]), dim=1), dims=[1]).to(torch.bool)  # (K,T,D)
    start_ok = near0 & suffix_all  # (K,T,D)

    any_true = start_ok.any(dim=1)  # (K,D)
    start_idx = start_ok.to(torch.float32).argmax(dim=1)  # (K,D), 0 if none
    start_idx = torch.where(any_true, start_idx, torch.full_like(start_idx, fill_value=T))

    t_grid = torch.arange(T, device=real_unnorm_ktd.device).view(1, T, 1)  # (1,T,1)
    return t_grid >= start_idx.view(K, 1, D)  # (K,T,D)


def apply_mask_nan_ktd(x_ktd: torch.Tensor, mask_ktd: torch.Tensor) -> torch.Tensor:
    if x_ktd.shape != mask_ktd.shape:
        raise ValueError(f"Shape mismatch: x={tuple(x_ktd.shape)} mask={tuple(mask_ktd.shape)}")
    nan = torch.tensor(float("nan"), device=x_ktd.device, dtype=x_ktd.dtype)
    return torch.where(mask_ktd, nan, x_ktd)


# =========================================================
# NEW: Conditional masking for synth
#   Only mask synth at real's zero-tail timesteps if that synth sample also has a near-zero tail.
# =========================================================
def compute_synth_has_zero_tail_ksd(synth_unnorm_ksdt: torch.Tensor, atol: float) -> torch.Tensor:
    """
    Returns has_tail_ksd (K,S,D) True if each synthetic trajectory has a near-zero suffix.
    """
    if synth_unnorm_ksdt.ndim != 4:
        raise ValueError(f"Expected synth (K,S,D,T), got {tuple(synth_unnorm_ksdt.shape)}")

    zero = torch.zeros((), device=synth_unnorm_ksdt.device, dtype=synth_unnorm_ksdt.dtype)
    finite = torch.isfinite(synth_unnorm_ksdt)
    near0 = finite & torch.isclose(synth_unnorm_ksdt, zero, atol=float(atol), rtol=0.0)  # (K,S,D,T)

    near0_i = near0.to(torch.int32)
    suffix_all = torch.flip(torch.cumprod(torch.flip(near0_i, dims=[-1]), dim=-1), dims=[-1]).to(torch.bool)  # (K,S,D,T)
    start_ok = near0 & suffix_all  # (K,S,D,T)
    return start_ok.any(dim=-1)  # (K,S,D)


def apply_real_tail_mask_nan_ksdt_if_synth_tail_zero(
    synth_unnorm_ksdt: torch.Tensor,
    real_mask_ktd: torch.Tensor,
    synth_atol: float,
) -> torch.Tensor:
    """
    Mask synth (K,S,D,T) with NaNs only where:
      - real has a zero tail (real_mask_ktd True), AND
      - that synth sample has a near-zero tail (within synth_atol)

    This prevents "zero-padded" synthetic tails from affecting CRPS/plots while keeping non-zero continuations.
    """
    if synth_unnorm_ksdt.ndim != 4:
        raise ValueError(f"Expected synth (K,S,D,T), got {tuple(synth_unnorm_ksdt.shape)}")
    if real_mask_ktd.ndim != 3:
        raise ValueError(f"Expected real_mask (K,T,D), got {tuple(real_mask_ktd.shape)}")

    K, S, D, T = synth_unnorm_ksdt.shape
    if real_mask_ktd.shape != (K, T, D):
        raise ValueError(f"Mask shape mismatch: mask={tuple(real_mask_ktd.shape)} expected={(K, T, D)}")

    has_tail_ksd = compute_synth_has_zero_tail_ksd(synth_unnorm_ksdt, atol=synth_atol)  # (K,S,D)

    real_mask_ksdt = real_mask_ktd.permute(0, 2, 1).unsqueeze(1).expand(K, S, D, T)  # (K,S,D,T)
    tail_gate_ksdt = has_tail_ksd.unsqueeze(-1).expand(K, S, D, T)  # (K,S,D,T)
    mask = real_mask_ksdt & tail_gate_ksdt

    nan = torch.tensor(float("nan"), device=synth_unnorm_ksdt.device, dtype=synth_unnorm_ksdt.dtype)
    return torch.where(mask, nan, synth_unnorm_ksdt)


# =========================================================
# DIAGNOSTIC: Check for near-zero values
# =========================================================
def diagnose_zero_behavior(
    real_unnorm_ktd: torch.Tensor,
    synth_unnorm_ksdt: torch.Tensor,
    var_names: List[str],
    threshold: float = 0.01,
) -> None:
    """
    Diagnose if synthetic data has small values where real has zeros.
    """
    print("\n" + "=" * 70)
    print("ZERO-VALUE DIAGNOSTIC")
    print("=" * 70)

    K, T, D = real_unnorm_ktd.shape

    for d in range(D):
        real_vals = real_unnorm_ktd[:, :, d]
        real_finite = real_vals[torch.isfinite(real_vals)]
        real_near_zero = (real_finite.abs() < threshold).sum().item()
        real_exact_zero = (real_finite == 0.0).sum().item()

        synth_vals = synth_unnorm_ksdt[:, :, d, :]
        synth_finite = synth_vals[torch.isfinite(synth_vals)]
        synth_near_zero = (synth_finite.abs() < threshold).sum().item()
        synth_exact_zero = (synth_finite == 0.0).sum().item()

        print(f"\n{var_names[d]}:")
        print(
            f"  Real:   exact_zeros={real_exact_zero:6d}, "
            f"near_zeros(<{threshold})={real_near_zero:6d}, "
            f"total_finite={real_finite.numel():6d}"
        )
        print(
            f"  Synth:  exact_zeros={synth_exact_zero:6d}, "
            f"near_zeros(<{threshold})={synth_near_zero:6d}, "
            f"total_finite={synth_finite.numel():6d}"
        )

        if real_near_zero > 0 and synth_near_zero > real_near_zero * 2:
            print(f"  ⚠️  Synthetic has {synth_near_zero / real_near_zero:.1f}x more near-zeros!")


# =========================================================
# UNCONDITIONAL ENSEMBLES
# =========================================================
def build_unconditional_ensembles_from_total(
    synth_norm_uncond: torch.Tensor,
    K_target: int,
    x: int,
    d_expected: int,
    seed: int = 0,
    allow_wrap: bool = True,
) -> torch.Tensor:
    """
    From unconditional pool, build (K_target, x, D, T) in normalized space.
    Accepts synth_norm_uncond as:
      - (K_u,S_u,D,T) or (K_u,S_u,T,D)
      - (N,D,T) or (N,T,D)
    """
    if synth_norm_uncond.ndim == 4:
        synth = ensure_synth_ksdt(synth_norm_uncond, d_expected=d_expected)
        K_u, S_u, D, T = synth.shape
        pool = synth.reshape(K_u * S_u, D, T)
    elif synth_norm_uncond.ndim == 3:
        pool = synth_norm_uncond
        if pool.shape[1] == d_expected:  # (N,D,T)
            D, T = pool.shape[1], pool.shape[2]
        elif pool.shape[2] == d_expected:  # (N,T,D)->(N,D,T)
            pool = pool.permute(0, 2, 1).contiguous()
            D, T = pool.shape[1], pool.shape[2]
        else:
            raise ValueError(f"Cannot infer unconditional layout from {tuple(pool.shape)}")
    else:
        raise ValueError(f"Expected 3D/4D unconditional tensor, got {tuple(synth_norm_uncond.shape)}")

    N = pool.shape[0]
    need = K_target * x
    if N == 0:
        raise ValueError("Unconditional pool is empty.")

    g = torch.Generator(device=pool.device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=pool.device)

    if N >= need:
        chosen = pool[perm[:need]]
    else:
        if not allow_wrap:
            raise ValueError(f"Not enough unconditional samples: have N={N}, need {need}")
        reps = (need + N - 1) // N
        chosen = pool[perm.repeat(reps)[:need]]

    return chosen.reshape(K_target, x, D, T)  # (K_target,S,D,T) with S=x


# =========================================================
# LOAD SAMPLES
# =========================================================
def load_generated_samples(
    samples_dir: str,
    var_names: List[str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Returns:
      real_unnorm_ref: (K,T,D) UNNORMALIZED
      synth_by_model_norm: model->(K,S,D,T) NORMALIZED
      scalers_by_model: model->(mu,sd)
    """
    d_expected = len(var_names)
    print(f"Loading generated samples from: {samples_dir}")

    name_map = {
        "combined": "Combined",
        "vae_only": "VAE only",
        "2d_traits": "2D traits",
        "unconditional": "Unconditional",
    }

    synth_by_model_norm: Dict[str, torch.Tensor] = {}
    scalers_by_model: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    real_raw: Optional[torch.Tensor] = None
    mu_ref: Optional[torch.Tensor] = None
    sd_ref: Optional[torch.Tensor] = None

    for filename in os.listdir(samples_dir):
        if not filename.endswith("_synthetic.pt"):
            continue

        model_key = filename.replace("_synthetic.pt", "")
        model_name = name_map.get(model_key, model_key)

        synth_path = os.path.join(samples_dir, f"{model_key}_synthetic.pt")
        synth_norm = torch.load(synth_path, map_location="cpu").float()
        if synth_norm.ndim == 4:
            synth_norm = ensure_synth_ksdt(synth_norm, d_expected=d_expected)

        norm_path = os.path.join(samples_dir, f"{model_key}_normalization.pt")
        norm_data = torch.load(norm_path, map_location="cpu")
        mu = norm_data["mean"].float()
        sd = norm_data["std"].float()

        synth_by_model_norm[model_name] = synth_norm
        scalers_by_model[model_name] = (mu, sd)

        if mu_ref is None:
            mu_ref, sd_ref = mu, sd
            print(f"Using {model_name} normalization as initial reference")

        if real_raw is None and model_key != "unconditional":
            real_path = os.path.join(samples_dir, f"{model_key}_real.pt")
            if os.path.exists(real_path):
                real_raw = torch.load(real_path, map_location="cpu").float()

    if real_raw is None:
        raise FileNotFoundError("Could not find real data file (*_real.pt)")

    real_raw = ensure_real_ktd(real_raw, d_expected=d_expected)

    assert mu_ref is not None and sd_ref is not None

    real_valid = real_raw[torch.isfinite(real_raw)]
    real_mean = real_valid.mean().item()
    real_std = real_valid.std().item()
    is_normalized = abs(real_mean) < 0.5 and 0.5 < real_std < 1.5

    if is_normalized:
        print("Real appears NORMALIZED -> unnormalizing with reference params")
        real_unnorm_ref = unnormalize(real_raw, mu_ref, sd_ref)
    else:
        print("Real appears UNNORMALIZED")
        real_unnorm_ref = real_raw

    print(f"Loaded {len(synth_by_model_norm)} models.")
    print(f"Real shape (K,T,D): {tuple(real_unnorm_ref.shape)}")
    for n, x in synth_by_model_norm.items():
        print(f"  {n}: synth shape {tuple(x.shape)}")
    return real_unnorm_ref, synth_by_model_norm, scalers_by_model


# =========================================================
# METRICS
# =========================================================
def _mean_ci_over_conditions(x_ktd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_t = np.nanmean(x_ktd, axis=0)
    std_t = np.nanstd(x_ktd, axis=0)
    n_t = np.sum(np.isfinite(x_ktd), axis=0).astype(np.float64)
    se_t = np.where(n_t > 0, std_t / np.sqrt(np.maximum(n_t, 1.0)), np.nan)
    lo = mean_t - MEAN_CI_Z * se_t
    hi = mean_t + MEAN_CI_Z * se_t
    return mean_t, lo, hi


def _mean_abs_pairwise_diff_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute E|X - X'| for samples x (1D) in O(n log n), ignoring NaNs.
    Returns NaN if <2 finite samples.
    """
    v = x[torch.isfinite(x)]
    n = v.numel()
    if n < 2:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    v, _ = torch.sort(v)
    idx = torch.arange(n, device=v.device, dtype=v.dtype)
    coeff = 2.0 * idx - float(n) + 1.0
    s = torch.sum(coeff * v)
    return (2.0 / (float(n) * float(n))) * s


def compute_crps_gpu(synth_ksdt: torch.Tensor, real_ktd: torch.Tensor) -> np.ndarray:
    """
    synth_ksdt: (K,S,D,T) normalized, may contain NaNs
    real_ktd: (K,T,D) normalized, may contain NaNs
    Returns: (K,T,D)
    """
    K, S, D, T = synth_ksdt.shape
    K2, T2, D2 = real_ktd.shape
    if (K2, T2, D2) != (K, T, D):
        raise ValueError(f"Shape mismatch: synth={tuple(synth_ksdt.shape)} real={tuple(real_ktd.shape)}")

    out = np.full((K, T, D), np.nan, dtype=np.float64)

    for k in tqdm(range(K), desc="Computing CRPS"):
        synth_k = synth_ksdt[k].to(DEVICE)  # (S,D,T)
        real_k = real_ktd[k].to(DEVICE)  # (T,D)

        real_dt = real_k.permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,D,T)
        diff = torch.abs(synth_k.unsqueeze(0) - real_dt)  # (1,S,D,T)

        real_nan = torch.isnan(real_dt)
        nan = torch.tensor(float("nan"), device=DEVICE, dtype=diff.dtype)
        diff = torch.where(real_nan.expand_as(diff), nan, diff)
        term1 = torch.nanmean(diff, dim=1).squeeze(0)  # (D,T)

        term2 = torch.empty((D, T), device=DEVICE, dtype=synth_k.dtype)
        for d in range(D):
            for t in range(T):
                term2[d, t] = _mean_abs_pairwise_diff_1d(synth_k[:, d, t])

        crps_dt = term1 - 0.5 * term2
        out[k] = crps_dt.permute(1, 0).detach().cpu().numpy()

    return out


def compute_variance_over_samples_gpu(synth_ksdt: torch.Tensor, batch_size: int = 50) -> np.ndarray:
    """
    synth_ksdt: (K,S,D,T) normalized (masked with NaNs)
    Returns: (K,T,D)
    """
    K, S, D, T = synth_ksdt.shape
    out = np.zeros((K, T, D), dtype=np.float64)

    for k0 in range(0, K, batch_size):
        k1 = min(k0 + batch_size, K)
        batch = synth_ksdt[k0:k1].to(DEVICE)  # (B,S,D,T)
        valid = torch.isfinite(batch)
        n = valid.sum(dim=1).float()  # (B,D,T)

        batch0 = torch.where(valid, batch, torch.zeros_like(batch))
        mean = batch0.sum(dim=1) / torch.clamp(n, min=1.0)  # (B,D,T)

        diff2 = (batch0 - mean.unsqueeze(1)) ** 2
        diff2 = torch.where(valid, diff2, torch.zeros_like(diff2))
        var = diff2.sum(dim=1) / torch.clamp(n - 1.0, min=1.0)  # (B,D,T)

        nan = torch.tensor(float("nan"), device=DEVICE, dtype=var.dtype)
        var = torch.where(n > 1.0, var, nan)
        out[k0:k1] = var.permute(0, 2, 1).detach().cpu().numpy()

    return out


def _safe_hist_prob(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.full((len(bins) - 1,), 1.0 / (len(bins) - 1), dtype=np.float64)
    counts, _ = np.histogram(vals, bins=bins)
    p = counts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / p.size)
    return p / s


def jsd_from_samples(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")

    lo = np.nanpercentile(np.concatenate([a, b]), 1)
    hi = np.nanpercentile(np.concatenate([a, b]), 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan")

    edges = np.linspace(lo, hi, bins + 1)
    p = _safe_hist_prob(a, edges)
    q = _safe_hist_prob(b, edges)

    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def compute_jsd_per_variable(
    real_unnorm_ktd: torch.Tensor,  # (K,T,D) unnormalized masked
    synth_norm_ref_ksdt: torch.Tensor,  # (K,S,D,T) normalized masked (reference space)
    mu_ref: torch.Tensor,
    sd_ref: torch.Tensor,
    var_names: List[str],
) -> Dict[str, float]:
    synth_unnorm = unnormalize(synth_norm_ref_ksdt, mu_ref, sd_ref)  # (K,S,D,T)
    out: Dict[str, float] = {}
    _, _, D = real_unnorm_ktd.shape
    for d in range(D):
        real_vals = real_unnorm_ktd[:, :, d].detach().cpu().numpy().reshape(-1)
        gen_vals = synth_unnorm[:, :, d, :].detach().cpu().numpy().reshape(-1)
        out[var_names[d]] = jsd_from_samples(gen_vals, real_vals, bins=200)
    return out


# =========================================================
# PLOTS
# =========================================================
def _kde1d(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.zeros_like(grid)
    std = np.std(data)
    if std == 0:
        dens = np.zeros_like(grid)
        dens[np.argmin(np.abs(grid - data.mean()))] = 1.0
        return dens
    n = data.size
    bw = 1.06 * std * (n ** (-1.0 / 5.0))
    diffs = (grid[None, :] - data[:, None]) / bw
    kern = np.exp(-0.5 * diffs**2) / (math.sqrt(2 * math.pi) * bw)
    return kern.mean(axis=0)


def plot_kde_per_variable_all_models(
    real_unnorm_ktd: torch.Tensor,
    synth_by_model_norm_ref_ksdt: Dict[str, torch.Tensor],
    var_names: List[str],
    outdir: str,
    color_map: Dict[str, str],
    mu_ref: torch.Tensor,
    sd_ref: torch.Tensor,
    num_grid: int = 256,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    _, _, D = real_unnorm_ktd.shape

    for d in range(D):
        real_vals = real_unnorm_ktd[:, :, d].detach().cpu().numpy().reshape(-1)
        finite_real = real_vals[np.isfinite(real_vals)]
        if finite_real.size == 0:
            continue

        vmin, vmax = np.nanpercentile(finite_real, [1, 99])
        pooled: Dict[str, np.ndarray] = {}

        for name, synth_norm_ref in synth_by_model_norm_ref_ksdt.items():
            synth_unnorm = unnormalize(synth_norm_ref, mu_ref, sd_ref)
            vals = synth_unnorm[:, :, d, :].detach().cpu().numpy().reshape(-1)
            pooled[name] = vals
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = np.nanpercentile(finite, [1, 99])
                vmin = min(vmin, lo)
                vmax = max(vmax, hi)

        grid = np.linspace(vmin, vmax, num_grid)
        dens_real = _kde1d(real_vals, grid)

        plt.figure(figsize=(10, 5))
        plt.plot(grid, dens_real, label="Real", linewidth=2, color=COLOR_REAL)

        for name, vals in pooled.items():
            plt.plot(grid, _kde1d(vals, grid), label=name, color=color_map.get(name, None))

        plt.title(f"KDE — {var_names[d]}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"kde_{var_names[d]}.png"))
        plt.close()


def plot_time_series_mean_ci_per_variable(
    series_by_model: Dict[str, np.ndarray],  # model->(K,T,D)
    var_names: List[str],
    outdir: str,
    title_prefix: str,
    ylabel: str,
    color_map: Dict[str, str],
    fname_prefix: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    example = next(iter(series_by_model.values()))
    _, T, D = example.shape
    x = np.arange(T)

    for d in range(D):
        plt.figure(figsize=(12, 6))
        for name, arr in series_by_model.items():
            x_kt = arr[:, :, d]
            mean_t, lo, hi = _mean_ci_over_conditions(x_kt)
            plt.plot(x, mean_t, linewidth=2, label=name, color=color_map.get(name, None))
            plt.fill_between(x, lo, hi, alpha=0.20, color=color_map.get(name, None))
        plt.xlabel("Time Step")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} — {var_names[d]} (mean ± 95% CI)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fname_prefix}_{var_names[d]}.png"))
        plt.close()


def plot_crps_over_time_no_ci(
    crps_norm_by_model: Dict[str, np.ndarray],
    var_names: List[str],
    outdir: str,
    color_map: Dict[str, str],
) -> None:
    os.makedirs(outdir, exist_ok=True)
    example = next(iter(crps_norm_by_model.values()))
    _, T, D = example.shape
    x = np.arange(T)

    for d in range(D):
        plt.figure(figsize=(12, 6))
        for name, crps_ktd in crps_norm_by_model.items():
            crps_td = crps_ktd[:, :, d]
            mean_t = np.nanmean(crps_td, axis=0)
            plt.plot(x, mean_t, linewidth=2.5, label=name, color=color_map.get(name, None))
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("CRPS (normalized)", fontsize=12)
        plt.title(f"CRPS Over Time — {var_names[d]}", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"crps_time_no_ci_{var_names[d]}.png"), dpi=150)
        plt.close()


def plot_crps_comparison_barplot(
    crps_norm_by_model: Dict[str, np.ndarray],
    var_names: List[str],
    outdir: str,
    conditional_key: str = "Combined",
    unconditional_key: str = "Unconditional",
) -> None:
    os.makedirs(outdir, exist_ok=True)

    if conditional_key not in crps_norm_by_model or unconditional_key not in crps_norm_by_model:
        print(f"⚠️  Cannot create comparison: need both {conditional_key} and {unconditional_key}")
        return

    crps_cond = crps_norm_by_model[conditional_key]
    crps_uncond = crps_norm_by_model[unconditional_key]

    overall_cond = np.nanmean(crps_cond)
    overall_uncond = np.nanmean(crps_uncond)

    plt.figure(figsize=(8, 6))
    x = np.array([0, 1])
    heights = [overall_cond, overall_uncond]
    colors_list = [COLOR_MAP_DEFAULT[conditional_key], COLOR_MAP_DEFAULT[unconditional_key]]

    bars = plt.bar(x, heights, width=0.6, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)

    for bar, height in zip(bars, heights):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.xticks(x, [conditional_key, unconditional_key], fontsize=12)
    plt.ylabel("Average CRPS (normalized)", fontsize=12)
    plt.title("Overall CRPS Comparison", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "crps_comparison_overall.png"), dpi=150)
    plt.close()

    K, T, D = crps_cond.shape
    per_var_cond = [np.nanmean(crps_cond[:, :, d]) for d in range(D)]
    per_var_uncond = [np.nanmean(crps_uncond[:, :, d]) for d in range(D)]

    x = np.arange(D)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width / 2,
        per_var_cond,
        width,
        label=conditional_key,
        color=COLOR_MAP_DEFAULT[conditional_key],
        alpha=0.8,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        per_var_uncond,
        width,
        label=unconditional_key,
        color=COLOR_MAP_DEFAULT[unconditional_key],
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_xlabel("Variable", fontsize=12)
    ax.set_ylabel("Average CRPS (normalized)", fontsize=12)
    ax.set_title("CRPS Comparison by Variable", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "crps_comparison_per_variable.png"), dpi=150)
    plt.close()

    improvement = []
    for d in range(D):
        cond_val = per_var_cond[d]
        uncond_val = per_var_uncond[d]
        if not np.isnan(uncond_val) and uncond_val != 0:
            improvement.append(((uncond_val - cond_val) / uncond_val) * 100)
        else:
            improvement.append(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["green" if imp > 0 else "red" for imp in improvement]
    ax.bar(x, improvement, color=colors, alpha=0.7, edgecolor="black")

    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.set_xlabel("Variable", fontsize=12)
    ax.set_ylabel("Improvement (%)", fontsize=12)
    ax.set_title(
        "CRPS Improvement: Conditional vs Unconditional\n(Positive = Conditional Better)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "crps_improvement_percentage.png"), dpi=150)
    plt.close()

    print(f"\n{'=' * 70}")
    print("CRPS COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print("Overall Average:")
    print(f"  {conditional_key:20s}: {overall_cond:.6f}")
    print(f"  {unconditional_key:20s}: {overall_uncond:.6f}")
    if overall_uncond != 0 and np.isfinite(overall_uncond):
        print(f"  Improvement: {((overall_uncond - overall_cond) / overall_uncond * 100):.2f}%")

    print("\nPer-Variable Improvement:")
    for d, var_name in enumerate(var_names):
        print(f"  {var_name:12s}: {improvement[d]:+6.2f}%")


def plot_overall_boxplot(metric_by_model: Dict[str, np.ndarray], outpath: str, title: str, ylabel: str) -> None:
    model_names = list(metric_by_model.keys())
    data = []
    for name in model_names:
        vals = metric_by_model[name]
        vals = vals[np.isfinite(vals)]
        data.append(vals)

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=model_names, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_per_variable_boxplots(
    metric_by_model: Dict[str, np.ndarray],
    var_names: List[str],
    outdir: str,
    title_prefix: str,
    ylabel: str,
    fname_prefix: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    example = next(iter(metric_by_model.values()))
    _, _, D = example.shape

    for name, arr in metric_by_model.items():
        fig = plt.figure(figsize=(max(10, D * 0.7), 6))
        data: List[np.ndarray] = []
        for d in range(D):
            vals = arr[:, :, d].ravel()
            vals = vals[np.isfinite(vals)]
            data.append(vals)

        plt.boxplot(data, showfliers=False)
        plt.xticks(np.arange(1, D + 1), var_names, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} — {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fname_prefix}_{name}.png"))
        plt.close(fig)


def plot_variance_over_time(
    var_by_model: Dict[str, np.ndarray],
    var_names: List[str],
    outdir: str,
    color_map: Dict[str, str],
    fname_prefix: str,
    title_prefix: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    example = next(iter(var_by_model.values()))
    _, T, D = example.shape
    x = np.arange(T)

    for d in range(D):
        plt.figure(figsize=(12, 6))
        for name, arr in var_by_model.items():
            x_kt = arr[:, :, d]
            mean_t, lo, hi = _mean_ci_over_conditions(x_kt)
            plt.plot(x, mean_t, linewidth=2, label=name, color=color_map.get(name, None))
            plt.fill_between(x, lo, hi, alpha=0.20, color=color_map.get(name, None))
        plt.title(f"{title_prefix} — {var_names[d]} (mean ± 95% CI)")
        plt.xlabel("Time Step")
        plt.ylabel("Variance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fname_prefix}_{var_names[d]}.png"))
        plt.close()


def _select_condition_indices(K: int) -> List[int]:
    if K <= 0:
        return []
    m = min(MAX_PLOT_CONDITIONS, K)
    idxs = np.linspace(0, K - 1, m).round().astype(int).tolist()
    return sorted(set(idxs))


def select_best_crps_conditions(
    crps_norm_ktd: np.ndarray,
    n_select: int = None,
) -> List[int]:
    if n_select is None:
        n_select = MAX_PLOT_CONDITIONS

    K, _, _ = crps_norm_ktd.shape
    crps_per_condition = np.nanmean(crps_norm_ktd, axis=(1, 2))
    sorted_indices = np.argsort(crps_per_condition)

    n_select = min(n_select, K)
    best_indices = sorted_indices[:n_select].tolist()

    print(f"\n{'=' * 70}")
    print(f"SELECTED BEST {n_select} CONDITIONS BY CRPS")
    print(f"{'=' * 70}")
    for rank, k in enumerate(best_indices, 1):
        crps_val = crps_per_condition[k]
        print(f"  Rank {rank:2d}: Condition {k:4d} with CRPS = {crps_val:.6f}")

    return best_indices


def plot_trajectories_ci_cond_vs_uncond_direct(
    real_unnorm_ktd: torch.Tensor,
    synth_cond_unnorm_ksdt: torch.Tensor,
    synth_uncond_unnorm_ksdt: torch.Tensor,
    var_names: List[str],
    outdir: str,
    crps_norm_ktd: Optional[np.ndarray] = None,
    n_plot: Optional[int] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    K, T, D = real_unnorm_ktd.shape

    if n_plot is None:
        n_plot = MAX_PLOT_CONDITIONS

    if crps_norm_ktd is not None:
        print("Selecting best CRPS conditions for trajectory plots...")
        cond_idxs = select_best_crps_conditions(crps_norm_ktd, n_select=n_plot)
    else:
        print("Using default condition selection (uniform spacing)...")
        cond_idxs = _select_condition_indices(K)

    x = np.arange(T)

    for k in tqdm(cond_idxs, desc="Trajectory plots (cond vs uncond)"):
        for d in range(D):
            plt.figure(figsize=(11, 6))
            real = real_unnorm_ktd[k, :, d].detach().cpu().numpy()
            plt.plot(x, real, linewidth=2, color=COLOR_REAL, label="Ground truth")

            s_c = synth_cond_unnorm_ksdt[k, :, d, :].detach().cpu().numpy()
            qlo_c = np.nanquantile(s_c, TRAJ_QUANTILES[0], axis=0)
            qhi_c = np.nanquantile(s_c, TRAJ_QUANTILES[1], axis=0)
            mean_c = np.nanmean(s_c, axis=0)
            plt.fill_between(
                x,
                qlo_c,
                qhi_c,
                alpha=0.25,
                color=COLOR_MAP_DEFAULT["Combined"],
                label="Conditional 95% interval",
            )
            plt.plot(x, mean_c, linewidth=2, color=COLOR_MAP_DEFAULT["Combined"], label="Conditional mean")

            s_u = synth_uncond_unnorm_ksdt[k, :, d, :].detach().cpu().numpy()
            qlo_u = np.nanquantile(s_u, TRAJ_QUANTILES[0], axis=0)
            qhi_u = np.nanquantile(s_u, TRAJ_QUANTILES[1], axis=0)
            mean_u = np.nanmean(s_u, axis=0)
            plt.fill_between(
                x,
                qlo_u,
                qhi_u,
                alpha=0.25,
                color=COLOR_MAP_DEFAULT["Unconditional"],
                label="Unconditional 95% interval",
            )
            plt.plot(x, mean_u, linewidth=2, color=COLOR_MAP_DEFAULT["Unconditional"], label="Unconditional mean")

            title = f"Condition Sample — {var_names[d]}"
            plt.title(title, fontsize=12, fontweight="bold")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(ncol=2, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"traj_ci_cond{k}_{var_names[d]}.png"))
            plt.close()


# =========================================================
# METRICS EXPORT
# =========================================================
def _summarize_metric(arr_ktd: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    arr_flat = arr_ktd[np.isfinite(arr_ktd)]
    _, _, D = arr_ktd.shape

    per_var_mean: Dict[str, float] = {}
    per_var_var: Dict[str, float] = {}
    for d in range(D):
        vals = arr_ktd[:, :, d].ravel()
        vals = vals[np.isfinite(vals)]
        per_var_mean[var_names[d]] = float(np.mean(vals)) if vals.size else float("nan")
        per_var_var[var_names[d]] = float(np.var(vals)) if vals.size else float("nan")

    return {
        "global": {
            "mean": float(np.mean(arr_flat)) if arr_flat.size else float("nan"),
            "variance": float(np.var(arr_flat)) if arr_flat.size else float("nan"),
        },
        "per_variable": {"mean": per_var_mean, "variance": per_var_var},
    }


def export_metrics_json(
    outpath: str,
    var_names: List[str],
    crps_norm_by_model: Dict[str, np.ndarray],
    crps_unnorm_by_model: Dict[str, np.ndarray],
    var_norm_by_model: Dict[str, np.ndarray],
    var_unnorm_by_model: Dict[str, np.ndarray],
    jsd_by_model: Dict[str, Dict[str, float]],
) -> None:
    stats: Dict[str, Any] = {
        "crps": {
            "normalized": {m: _summarize_metric(a, var_names) for m, a in crps_norm_by_model.items()},
            "unnormalized": {m: _summarize_metric(a, var_names) for m, a in crps_unnorm_by_model.items()},
        },
        "variance": {
            "normalized": {m: _summarize_metric(a, var_names) for m, a in var_norm_by_model.items()},
            "unnormalized": {m: _summarize_metric(a, var_names) for m, a in var_unnorm_by_model.items()},
        },
        "jsd": jsd_by_model,
    }
    with open(outpath, "w") as f:
        json.dump(stats, f, indent=4)


# =========================================================
# ANALYSIS SUITE
# =========================================================
def run_analysis_suite(
    suite_name: str,
    out_root: str,
    var_names: List[str],
    real_unnorm_ref: torch.Tensor,
    synth_by_model_norm: Dict[str, torch.Tensor],
    scalers_by_model: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    color_map: Dict[str, str],
    make_cond_vs_uncond_trajectories: bool = False,
) -> None:
    os.makedirs(out_root, exist_ok=True)
    d_expected = len(var_names)

    mu_ref, sd_ref = scalers_by_model.get("Combined", next(iter(scalers_by_model.values())))

    real_unnorm_ref = ensure_real_ktd(real_unnorm_ref, d_expected=d_expected)
    synth_std: Dict[str, torch.Tensor] = {n: ensure_synth_ksdt(x, d_expected=d_expected) for n, x in synth_by_model_norm.items()}

    K_min = real_unnorm_ref.shape[0]
    for name, x in synth_std.items():
        if name != "Unconditional":
            K_min = min(K_min, x.shape[0])

    K_eval = min(K_min, MAX_EVAL_CONDITIONS) if MAX_EVAL_CONDITIONS else K_min

    print(f"\n{'=' * 70}")
    print(f"Running suite: {suite_name}")
    print(f"K_min={K_min}, K_eval={K_eval} (MAX_EVAL_CONDITIONS={MAX_EVAL_CONDITIONS})")
    print(f"ZERO_TAIL_ATOL={ZERO_TAIL_ATOL} | SYNTH_ZERO_TAIL_ATOL={SYNTH_ZERO_TAIL_ATOL} | MAX_PLOT_CONDITIONS={MAX_PLOT_CONDITIONS}")
    print(f"{'=' * 70}\n")

    real_unnorm_eval = real_unnorm_ref[:K_eval].clone()

    mask_ktd = compute_real_zero_tail_mask_ktd(real_unnorm_eval, atol=ZERO_TAIL_ATOL)
    real_unnorm_eval_masked = apply_mask_nan_ktd(real_unnorm_eval, mask_ktd)
    real_norm_ref_masked = normalize_from_unnorm(real_unnorm_eval_masked, mu_ref, sd_ref)

    synth_by_model_unnorm_masked: Dict[str, torch.Tensor] = {}
    synth_by_model_norm_ref_masked: Dict[str, torch.Tensor] = {}

    for name, synth_norm in synth_std.items():
        mu, sd = scalers_by_model[name]

        if name == "Unconditional":
            S_model = synth_norm.shape[1]
            synth_norm_trimmed = build_unconditional_ensembles_from_total(
                synth_norm_uncond=synth_norm,
                K_target=K_eval,
                x=S_model,
                d_expected=d_expected,
                seed=0,
                allow_wrap=True,
            )
        else:
            synth_norm_trimmed = synth_norm[:K_eval]

        synth_unnorm = unnormalize(synth_norm_trimmed, mu, sd)

        # NEW: conditional masking only for synth samples that have near-zero tails
        synth_unnorm_masked = apply_real_tail_mask_nan_ksdt_if_synth_tail_zero(
            synth_unnorm_ksdt=synth_unnorm,
            real_mask_ktd=mask_ktd,
            synth_atol=SYNTH_ZERO_TAIL_ATOL,
        )

        synth_norm_ref_masked = normalize_from_unnorm(synth_unnorm_masked, mu_ref, sd_ref)

        synth_by_model_unnorm_masked[name] = synth_unnorm_masked
        synth_by_model_norm_ref_masked[name] = synth_norm_ref_masked

    print("\n" + "=" * 70)
    print("RUNNING ZERO-VALUE DIAGNOSTICS")
    print("=" * 70)
    for name, synth_unnorm_masked in synth_by_model_unnorm_masked.items():
        print(f"\n{name}:")
        diagnose_zero_behavior(
            real_unnorm_eval_masked,
            synth_unnorm_masked,
            var_names,
            threshold=0.01,
        )

    crps_norm_by_model: Dict[str, np.ndarray] = {}
    crps_unnorm_by_model: Dict[str, np.ndarray] = {}
    var_norm_by_model: Dict[str, np.ndarray] = {}
    var_unnorm_by_model: Dict[str, np.ndarray] = {}
    jsd_by_model: Dict[str, Dict[str, float]] = {}

    sd_ref_np = sd_ref.view(-1).detach().cpu().numpy().reshape(1, 1, -1)

    for name, synth_norm_ref in synth_by_model_norm_ref_masked.items():
        print(f"Computing metrics for {name}...")

        crps_norm = compute_crps_gpu(synth_norm_ref, real_norm_ref_masked)
        crps_norm_by_model[name] = crps_norm
        crps_unnorm_by_model[name] = crps_norm * sd_ref_np

        v_norm = compute_variance_over_samples_gpu(synth_norm_ref)
        var_norm_by_model[name] = v_norm
        var_unnorm_by_model[name] = v_norm * (sd_ref_np ** 2)

        jsd_by_model[name] = compute_jsd_per_variable(
            real_unnorm_ktd=real_unnorm_eval_masked,
            synth_norm_ref_ksdt=synth_norm_ref,
            mu_ref=mu_ref,
            sd_ref=sd_ref,
            var_names=var_names,
        )

        print(f"  CRPS(norm)  mean={np.nanmean(crps_norm):.6f}")
        print(f"  CRPS(unn)   mean={np.nanmean(crps_unnorm_by_model[name]):.6f}")
        print()

    metrics_path = os.path.join(out_root, "metrics_summary.json")
    export_metrics_json(
        metrics_path,
        var_names,
        crps_norm_by_model,
        crps_unnorm_by_model,
        var_norm_by_model,
        var_unnorm_by_model,
        jsd_by_model,
    )

    crps_dir = os.path.join(out_root, "crps")
    os.makedirs(crps_dir, exist_ok=True)

    plot_per_variable_boxplots(
        crps_unnorm_by_model,
        var_names,
        outdir=crps_dir,
        title_prefix="CRPS (unnormalized) per variable",
        ylabel="CRPS",
        fname_prefix="crps_boxplots_unnorm",
    )

    plot_overall_boxplot(
        crps_unnorm_by_model,
        outpath=os.path.join(crps_dir, "crps_overall_boxplot_unnorm.png"),
        title=f"{suite_name}: CRPS (unnormalized)",
        ylabel="CRPS",
    )

    plot_time_series_mean_ci_per_variable(
        crps_norm_by_model,
        var_names,
        outdir=crps_dir,
        title_prefix="CRPS (normalized) over time",
        ylabel="CRPS (normalized)",
        color_map=color_map,
        fname_prefix="crps_time_mean_ci_norm",
    )

    print("Generating CRPS over time (no CI)...")
    plot_crps_over_time_no_ci(
        crps_norm_by_model,
        var_names,
        outdir=crps_dir,
        color_map=color_map,
    )

    if "Combined" in crps_norm_by_model and "Unconditional" in crps_norm_by_model:
        print("Generating CRPS comparison bar plots...")
        plot_crps_comparison_barplot(
            crps_norm_by_model,
            var_names,
            outdir=crps_dir,
            conditional_key="Combined",
            unconditional_key="Unconditional",
        )

    var_dir = os.path.join(out_root, "variance")
    os.makedirs(var_dir, exist_ok=True)

    plot_variance_over_time(
        var_unnorm_by_model,
        var_names,
        outdir=var_dir,
        color_map=color_map,
        fname_prefix="variance_time_mean_ci_unnorm",
        title_prefix="Variance (unnormalized) over time",
    )

    kde_dir = os.path.join(out_root, "kde")
    plot_kde_per_variable_all_models(
        real_unnorm_eval_masked,
        synth_by_model_norm_ref_masked,
        var_names,
        kde_dir,
        color_map=color_map,
        mu_ref=mu_ref,
        sd_ref=sd_ref,
    )

    if make_cond_vs_uncond_trajectories:
        if "Combined" in synth_by_model_unnorm_masked and "Unconditional" in synth_by_model_unnorm_masked:
            traj_dir = os.path.join(out_root, "trajectories_cond_vs_uncond")
            crps_cond = crps_norm_by_model.get("Combined", None)

            plot_trajectories_ci_cond_vs_uncond_direct(
                real_unnorm_eval_masked,
                synth_by_model_unnorm_masked["Combined"],
                synth_by_model_unnorm_masked["Unconditional"],
                var_names,
                traj_dir,
                crps_norm_ktd=crps_cond,
                n_plot=MAX_PLOT_CONDITIONS,
            )
        else:
            print("Skipping trajectories: need both Combined and Unconditional.")

    print(f"\n✓ Suite '{suite_name}' complete. Output: {out_root}\n")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    var_names = [
        "WRF_TEMP",
        "WRF_PRES",
        "WRF_RELH",
        "WRF_PHI",
        "WRF_PHIS",
        "WRF_QICE",
        "WRF_QSNOW",
        "WRF_QVAPOR",
        "WRF_QCLOUD",
        "WRF_QRAIN",
    ]
    MAX_EVAL_CONDITIONS = 2134
    samples_dir = "./synthetic_samples"
    out_base = "analysis_results_acc"

    real_unnorm_ref, synth_by_model_norm, scalers_by_model = load_generated_samples(samples_dir, var_names)

    run_analysis_suite(
        suite_name="All models",
        out_root=os.path.join(out_base, "all_4_models"),
        var_names=var_names,
        real_unnorm_ref=real_unnorm_ref,
        synth_by_model_norm=synth_by_model_norm,
        scalers_by_model=scalers_by_model,
        color_map=COLOR_MAP_DEFAULT,
        make_cond_vs_uncond_trajectories=False,
    )

    if "Combined" in synth_by_model_norm and "Unconditional" in synth_by_model_norm:
        suite_models = {
            "Unconditional": synth_by_model_norm["Unconditional"],
            "Combined": synth_by_model_norm["Combined"],
        }
        suite_scalers = {
            "Unconditional": scalers_by_model["Unconditional"],
            "Combined": scalers_by_model["Combined"],
        }
        suite_colors = {
            "Unconditional": COLOR_MAP_DEFAULT["Unconditional"],
            "Combined": COLOR_MAP_DEFAULT["Combined"],
        }

        run_analysis_suite(
            suite_name="Conditional vs Unconditional",
            out_root=os.path.join(out_base, "cond_vs_uncond_only"),
            var_names=var_names,
            real_unnorm_ref=real_unnorm_ref,
            synth_by_model_norm=suite_models,
            scalers_by_model=suite_scalers,
            color_map=suite_colors,
            make_cond_vs_uncond_trajectories=True,
        )

    print("\n" + "=" * 70)
    print("ALL ANALYSIS COMPLETE!")
    print("=" * 70)
