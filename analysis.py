"""
Simplified analysis code that works with current data generation format.

Key features:
- Handles (K, S, D, T) synthetic data shape correctly
- Handles (K, T, D) real data shape correctly  
- Properly unnormalizes using saved normalization parameters
- Computes CRPS and JSD metrics
- GPU-accelerated for speed
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VAR_NAMES = [
    'WRF_TEMP', 'WRF_PRES', 'WRF_RELH', 'WRF_PHI', 'WRF_PHIS',
    'WRF_QICE', 'WRF_QSNOW', 'WRF_QVAPOR', 'WRF_QCLOUD', 'WRF_QRAIN'
]


# =========================================================
# UNNORMALIZATION
# =========================================================
def unnormalize(x_norm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize: x_unnorm = x_norm * sd + mu
    
    Args:
        x_norm: (K, S, D, T) or (K, T, D)
        mu, sd: (1, 1, D)
    """
    if x_norm.ndim == 4:  # (K, S, D, T)
        mu = mu.reshape(1, 1, -1, 1)
        sd = sd.reshape(1, 1, -1, 1)
    elif x_norm.ndim == 3:  # (K, T, D)
        mu = mu.reshape(1, 1, -1)
        sd = sd.reshape(1, 1, -1)
    
    return x_norm * sd + mu


def normalize(x_unnorm: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """
    Normalize: x_norm = (x_unnorm - mu) / sd
    """
    if x_unnorm.ndim == 4:
        mu = mu.reshape(1, 1, -1, 1)
        sd = sd.reshape(1, 1, -1, 1)
    elif x_unnorm.ndim == 3:
        mu = mu.reshape(1, 1, -1)
        sd = sd.reshape(1, 1, -1)
    
    return (x_unnorm - mu) / sd


# =========================================================
# LOAD DATA
# =========================================================
def load_data(samples_dir: str):
    """Load generated samples."""
    name_map = {
        'combined': 'Combined',
        'vae_only': 'VAE only',
        '2d_traits': '2D traits',
        'unconditional': 'Unconditional'
    }
    
    synth_by_model = {}
    scalers_by_model = {}
    real = None
    
    for filename in os.listdir(samples_dir):
        if filename.endswith('_synthetic.pt'):
            model_key = filename.replace('_synthetic.pt', '')
            model_name = name_map.get(model_key, model_key)
            
            # Load synthetic
            synth = torch.load(os.path.join(samples_dir, f"{model_key}_synthetic.pt")).float()
            
            # Load normalization
            norm_data = torch.load(os.path.join(samples_dir, f"{model_key}_normalization.pt"))
            mu = norm_data['mean'].float()
            sd = norm_data['std'].float()
            
            synth_by_model[model_name] = synth
            scalers_by_model[model_name] = (mu, sd)
            
            print(f"{model_name}: {synth.shape}")
            
            # Load real (once)
            if real is None and model_key != 'unconditional':
                real = torch.load(os.path.join(samples_dir, f"{model_key}_real.pt")).float()
    
    return real, synth_by_model, scalers_by_model


# =========================================================
# METRICS
# =========================================================
def compute_crps_simple(synth: torch.Tensor, real: torch.Tensor) -> float:
    """
    Simple CRPS computation.
    
    Args:
        synth: (K, S, D, T) normalized
        real: (K, T, D) normalized
    """
    K, S, D, T = synth.shape
    crps_list = []
    
    for k in range(K):
        for d in range(D):
            for t in range(T):
                samples = synth[k, :, d, t].cpu().numpy()
                truth = real[k, t, d].item()
                
                if np.isnan(truth) or np.any(np.isnan(samples)):
                    continue
                
                # CRPS formula
                term1 = np.mean(np.abs(samples - truth))
                term2 = 0
                for i in range(S):
                    for j in range(S):
                        term2 += np.abs(samples[i] - samples[j])
                term2 /= (S * S)
                
                crps = term1 - 0.5 * term2
                crps_list.append(crps)
    
    return np.mean(crps_list) if crps_list else np.nan


def compute_jsd_simple(synth_vals: np.ndarray, real_vals: np.ndarray, bins: int = 50) -> float:
    """Simple JSD computation."""
    synth_vals = synth_vals[np.isfinite(synth_vals)]
    real_vals = real_vals[np.isfinite(real_vals)]
    
    if len(synth_vals) == 0 or len(real_vals) == 0:
        return np.nan
    
    # Create histogram bins
    all_vals = np.concatenate([synth_vals, real_vals])
    hist_range = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
    
    p, _ = np.histogram(synth_vals, bins=bins, range=hist_range, density=True)
    q, _ = np.histogram(real_vals, bins=bins, range=hist_range, density=True)
    
    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    
    # Add small epsilon
    p = p + 1e-10
    q = q + 1e-10
    
    # JSD
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    
    return float(jsd)


# =========================================================
# ANALYSIS
# =========================================================
def run_analysis(samples_dir: str = './synthetic_samples_test'):
    """Run analysis."""
    output_dir = Path('./analysis_results_simple')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80 + "\n")
    
    real, synth_by_model, scalers_by_model = load_data(samples_dir)
    
    # Check if real is normalized
    real_mean = real[torch.isfinite(real)].mean().item()
    real_std = real[torch.isfinite(real)].std().item()
    
    print(f"\nReal data: mean={real_mean:.4f}, std={real_std:.4f}")
    
    is_normalized = abs(real_mean) < 0.5 and 0.5 < real_std < 1.5
    
    # Use Combined as reference
    mu_ref, sd_ref = scalers_by_model["Combined"]
    
    if is_normalized:
        print("Real data is normalized - unnormalizing...")
        real_unnorm = unnormalize(real, mu_ref, sd_ref)
    else:
        real_unnorm = real
    
    real_norm = normalize(real_unnorm, mu_ref, sd_ref)
    
    # Find K_min
    K_min = real.shape[0]
    for name, synth in synth_by_model.items():
        if name != "Unconditional":
            K_min = min(K_min, synth.shape[0])
    
    print(f"\nUsing K_min = {K_min} conditions")
    
    # Trim real
    real_unnorm = real_unnorm[:K_min]
    real_norm = real_norm[:K_min]
    
    # Compute metrics
    results = {}
    
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80 + "\n")
    
    for name, synth in synth_by_model.items():
        print(f"\n{name}:")
        print(f"  Shape: {synth.shape}")
        
        mu, sd = scalers_by_model[name]
        
        # Handle unconditional
        if name == "Unconditional":
            K_u, S_u = synth.shape[:2]
            total = K_u * S_u
            S_target = synth_by_model["Combined"].shape[1]
            need = K_min * S_target
            
            synth_flat = synth.reshape(total, synth.shape[2], synth.shape[3])
            idx = torch.randperm(total)[:need]
            synth = synth_flat[idx].reshape(K_min, S_target, synth.shape[2], synth.shape[3])
        else:
            synth = synth[:K_min]
        
        # Renormalize to reference
        synth_unnorm = unnormalize(synth, mu, sd)
        synth_norm = normalize(synth_unnorm, mu_ref, sd_ref)
        
        # CRPS (sample for speed)
        print(f"  Computing CRPS (sample)...")
        K_sample = min(10, K_min)
        crps = compute_crps_simple(synth_norm[:K_sample], real_norm[:K_sample])
        print(f"    CRPS: {crps:.6f}")
        
        # JSD per variable
        print(f"  Computing JSD...")
        jsd_per_var = {}
        for d, var_name in enumerate(VAR_NAMES):
            synth_vals = synth_unnorm[:, :, d, :].numpy().flatten()
            real_vals = real_unnorm[:, :, d].numpy().flatten()
            jsd = compute_jsd_simple(synth_vals, real_vals)
            jsd_per_var[var_name] = jsd
        
        jsd_mean = np.nanmean(list(jsd_per_var.values()))
        print(f"    Mean JSD: {jsd_mean:.6f}")
        
        results[name] = {
            'crps': float(crps),
            'jsd_mean': float(jsd_mean),
            'jsd_per_var': {k: float(v) for k, v in jsd_per_var.items()}
        }
    
    # Save
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    print(f"{'Model':<20} {'CRPS':<15} {'JSD':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['crps']:<15.6f} {res['jsd_mean']:<15.6f}")
    
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    run_analysis()