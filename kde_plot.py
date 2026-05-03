"""
plot_kde_masked.py

Generates shaded KDE plots for Combined vs Unconditional,
masking out values in (-20, 20) to remove the near-zero hump.

Usage:
    python plot_kde_masked.py
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_DIR   = "./synthetic_samples"
OUT_DIR       = "./kde_masked_output"
ZERO_MASK_LO  = -10.0   # values in (ZERO_MASK_LO, ZERO_MASK_HI) are excluded
ZERO_MASK_HI  =  10.0
NUM_GRID      = 512
ALPHA_FILL    = 0.30

VAR_NAMES = [
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

COLOR_REAL  = "black"
COLOR_COND  = "tab:red"
COLOR_UNCOND = "tab:blue"

# ── Helpers ───────────────────────────────────────────────────────────────────

def unnormalize(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """x is (K,S,D,T); broadcast mu/sd over (1,1,D,1)."""
    return x * sd.reshape(1, 1, -1, 1) + mu.reshape(1, 1, -1, 1)


def _kde1d(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Silverman's-rule KDE, ignores non-finite values."""
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.zeros_like(grid)
    std = np.std(data)
    if std == 0:
        dens = np.zeros_like(grid)
        dens[np.argmin(np.abs(grid - data.mean()))] = 1.0
        return dens
    bw = 1.06 * std * (data.size ** (-0.2))
    diffs = (grid[None, :] - data[:, None]) / bw
    return np.exp(-0.5 * diffs ** 2).mean(axis=0) / (math.sqrt(2 * math.pi) * bw)


def mask_near_zero(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Replace values in (lo, hi) with NaN."""
    out = vals.copy().astype(np.float64)
    out[(out > lo) & (out < hi)] = np.nan
    return out


def load_model(samples_dir: str, key: str, d_expected: int):
    """Load synthetic (K,S,D,T), mu, sd for one model key."""
    synth = torch.load(
        os.path.join(samples_dir, f"{key}_synthetic.pt"), map_location="cpu"
    ).float()

    # ensure (K,S,D,T)
    if synth.ndim == 4:
        k, s, a, b = synth.shape
        if b == d_expected:          # (K,S,T,D) → (K,S,D,T)
            synth = synth.permute(0, 1, 3, 2).contiguous()
    else:
        raise ValueError(f"Unexpected synth shape: {tuple(synth.shape)}")

    norm = torch.load(
        os.path.join(samples_dir, f"{key}_normalization.pt"), map_location="cpu"
    )
    mu = norm["mean"].float()
    sd = norm["std"].float()
    return synth, mu, sd


def load_real(samples_dir: str, key: str, d_expected: int, mu: torch.Tensor, sd: torch.Tensor):
    """Load real data; auto-detect if normalized and unnormalize if needed."""
    real_path = os.path.join(samples_dir, f"{key}_real.pt")
    real = torch.load(real_path, map_location="cpu").float()

    # ensure (K,T,D)
    if real.ndim == 3:
        k, a, b = real.shape
        if b == d_expected:
            pass                     # already (K,T,D)
        elif a == d_expected:
            real = real.permute(0, 2, 1).contiguous()  # (K,D,T)→(K,T,D)

    finite = real[torch.isfinite(real)]
    if abs(finite.mean().item()) < 0.5 and 0.5 < finite.std().item() < 1.5:
        # looks normalized
        real = real * sd.reshape(1, 1, -1) + mu.reshape(1, 1, -1)

    return real  # (K,T,D) unnormalized


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    D = len(VAR_NAMES)

    print("Loading Combined …")
    synth_cond,   mu_cond,   sd_cond   = load_model(SAMPLES_DIR, "combined",      D)
    print("Loading Unconditional …")
    synth_uncond, mu_uncond, sd_uncond = load_model(SAMPLES_DIR, "unconditional", D)
    print("Loading real …")
    real_ktd = load_real(SAMPLES_DIR, "combined", D, mu_cond, sd_cond)

    # Unnormalize synthetic → (K,S,D,T)
    synth_cond_unnorm   = unnormalize(synth_cond,   mu_cond,   sd_cond)
    synth_uncond_unnorm = unnormalize(synth_uncond, mu_uncond, sd_uncond)

    for d, vname in enumerate(VAR_NAMES):
        print(f"  Plotting {vname} …")

        real_vals_raw   = real_ktd[:, :, d].numpy().ravel()
        cond_vals_raw   = synth_cond_unnorm[:, :, d, :].numpy().ravel()
        uncond_vals_raw = synth_uncond_unnorm[:, :, d, :].numpy().ravel()

        # Determine plot range from UNMASKED finite values
        all_finite = np.concatenate([
            real_vals_raw[np.isfinite(real_vals_raw)],
            cond_vals_raw[np.isfinite(cond_vals_raw)],
            uncond_vals_raw[np.isfinite(uncond_vals_raw)],
        ])

        # Apply near-zero mask only for KDE estimation
        real_vals   = mask_near_zero(real_vals_raw,   ZERO_MASK_LO, ZERO_MASK_HI)
        cond_vals   = mask_near_zero(cond_vals_raw,   ZERO_MASK_LO, ZERO_MASK_HI)
        uncond_vals = mask_near_zero(uncond_vals_raw, ZERO_MASK_LO, ZERO_MASK_HI)
        if all_finite.size == 0:
            print(f"    No finite data for {vname}, skipping.")
            continue

        vmin = np.percentile(all_finite, 0)
        vmax = np.percentile(all_finite, 100)
        grid = np.linspace(vmin, vmax, NUM_GRID)

        dens_real   = _kde1d(real_vals,   grid)
        dens_cond   = _kde1d(cond_vals,   grid)
        dens_uncond = _kde1d(uncond_vals, grid)

        fig, ax = plt.subplots(figsize=(9, 4))

        # Unconditional — shaded + line
        ax.fill_between(grid, dens_uncond, alpha=ALPHA_FILL, color=COLOR_UNCOND)
        ax.plot(grid, dens_uncond, color=COLOR_UNCOND, linewidth=1.8, label="Unconditional")

        # Conditional — shaded + line
        ax.fill_between(grid, dens_cond, alpha=ALPHA_FILL, color=COLOR_COND)
        ax.plot(grid, dens_cond, color=COLOR_COND, linewidth=1.8, label="Combined (conditional)")

        # Real — line only (black)
        ax.plot(grid, dens_real, color=COLOR_REAL, linewidth=2.0,
                linestyle="--", label="Real")

        ax.set_title(f"KDE — {vname}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        outpath = os.path.join(OUT_DIR, f"kde_masked_{vname}.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"    Saved → {outpath}")

    print(f"\nDone. Plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()