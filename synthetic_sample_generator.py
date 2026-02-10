import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from main_model import CSDI_PM25
from dataset_crystaltraj import get_dataloader


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config_path, ckpt_path, device='cuda'):
    """Load trained model from checkpoint."""
    config = load_config(config_path)
    model = CSDI_PM25(config, device=device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model, config


def generate_conditional_samples(model, test_loader, S=100, device='cuda', batch_size=32, max_conditions=None):
    """
    Generate S synthetic samples for each test condition.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        S: Number of samples per condition
        device: Device to run on
        batch_size: Batch size for generation
        max_conditions: If not None, only generate for first N conditions (for testing)
    
    """
    synthetic_list = []
    real_list = []
    conditioning_list = []
    
    K = len(test_loader.dataset)
    
    if max_conditions is not None:
        K_actual = min(K, max_conditions)
        print(f"Generating {S} samples for {K_actual}/{K} test conditions (SUBSET FOR TESTING)...")
    else:
        K_actual = K
        print(f"Generating {S} samples for each of {K} test conditions...")
    
    conditions_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing test batches")):

            if max_conditions is not None and conditions_processed >= max_conditions:
                break
            
            cond = batch['conditioning_data'].to(device)  # [B, D, T]
            real = batch['observed_data'].to(device)  # [B, D, T]
            
            B = cond.shape[0]
            
            for i in range(B):
                if max_conditions is not None and conditions_processed >= max_conditions:
                    break
                
                cond_single = cond[i:i+1]  # [1, D, T]
                real_single = real[i:i+1]  # [1, D, T]
                
                samples_for_this_condition = []
                num_batches = (S + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, S)
                    current_batch_size = batch_end - batch_start
                    
                    cond_repeated = cond_single.repeat(current_batch_size, 1, 1)
                    
                    synth_batch = model.synthesize(conditioning_data=cond_repeated)
                    samples_for_this_condition.append(synth_batch.cpu())
                
                synth = torch.cat(samples_for_this_condition, dim=0)  # [S, T, D]
                
                synthetic_list.append(synth.numpy())
                real_list.append(real_single.squeeze(0).permute(1, 0).cpu().numpy())  # [T, D]
                conditioning_list.append(cond_single.squeeze(0).permute(1, 0).cpu().numpy())
                
                conditions_processed += 1

    synthetic_samples = np.stack(synthetic_list, axis=0)  # [K_actual, S, T, D] 
    real_samples = np.stack(real_list, axis=0)  # [K_actual, T, D]
    conditioning = np.stack(conditioning_list, axis=0)  # [K_actual, T, cond_dim]
    
    print(f"Generated samples for {conditions_processed} conditions")
    
    return synthetic_samples, real_samples, conditioning


def generate_unconditional_samples(model, K=2144, S=100, device='cuda', batch_size=32):
    """
    Generate unconditional samples.
    
    Returns NORMALIZED samples (as output by the model).
    """
    synthetic_list = []
    
    total_samples = K * S
    print(f"Generating {total_samples} unconditional samples ({K} conditions x {S} samples)...")
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating unconditional samples"):
            batch_size_actual = min(batch_size, total_samples - batch_idx * batch_size)
            
            synth = model.synthesize(batch_size=batch_size_actual)
            synthetic_list.append(synth.cpu().numpy())
    
    all_samples = np.concatenate(synthetic_list, axis=0) 
    
    synthetic_samples = all_samples.reshape(K, S, all_samples.shape[1], all_samples.shape[2])
    
    return synthetic_samples


def save_results(output_dir, model_name, synthetic_samples, real_samples=None, 
                conditioning=None, target_mean=None, target_std=None):
    """
    Save generated samples and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving {model_name}...")
    print(f"    Synthetic shape: {synthetic_samples.shape} (NORMALIZED)")
    
    synth_valid = synthetic_samples[np.isfinite(synthetic_samples)]
    if synth_valid.size > 0:
        print(f"    Synthetic stats: mean={synth_valid.mean():.4f}, std={synth_valid.std():.4f}")
        print(f"    Synthetic range: [{synth_valid.min():.4f}, {synth_valid.max():.4f}]")
    else:
        print(f"    ⚠️  WARNING: All synthetic samples are NaN!")
    
    torch.save(
        torch.from_numpy(synthetic_samples),
        output_dir / f"{model_name}_synthetic.pt"
    )
    print(f"    ✓ Saved: {model_name}_synthetic.pt (NORMALIZED)")
    
    if real_samples is not None:
        print(f"    Real shape: {real_samples.shape} (NORMALIZED)")
        real_valid = real_samples[np.isfinite(real_samples)]
        if real_valid.size > 0:
            print(f"    Real stats: mean={real_valid.mean():.4f}, std={real_valid.std():.4f}")
            print(f"    Real range: [{real_valid.min():.4f}, {real_valid.max():.4f}]")
        
        torch.save(
            torch.from_numpy(real_samples),
            output_dir / f"{model_name}_real.pt"
        )
        print(f"    ✓ Saved: {model_name}_real.pt (NORMALIZED)")
    
    if conditioning is not None:
        torch.save(
            torch.from_numpy(conditioning),
            output_dir / f"{model_name}_conditioning.pt"
        )
        print(f"    ✓ Saved: {model_name}_conditioning.pt")
    
    if target_mean is not None and target_std is not None:
        if target_mean.ndim == 1:
            target_mean = target_mean.reshape(1, 1, -1)
            target_std = target_std.reshape(1, 1, -1)
        elif target_mean.ndim == 3 and target_mean.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected target_mean shape: {target_mean.shape}")
        
        print(f"    Normalization params:")
        print(f"      Shape: {target_mean.shape}")
        print(f"      Mean (first 3): {target_mean.flatten()[:3]}")
        print(f"      Std (first 3): {target_std.flatten()[:3]}")
        
        torch.save({
            'mean': torch.from_numpy(target_mean),
            'std': torch.from_numpy(target_std)
        }, output_dir / f"{model_name}_normalization.pt")
        print(f"    ✓ Saved: {model_name}_normalization.pt")
    
    metadata = {
        'shape': synthetic_samples.shape,
        'description': f"Shape: [K={synthetic_samples.shape[0]}, S={synthetic_samples.shape[1]}, T={synthetic_samples.shape[2]}, D={synthetic_samples.shape[3]}]",
        'data_state': 'NORMALIZED (mean~0, std~1)',
        'num_nans': int(np.isnan(synthetic_samples).sum()),
        'value_range': [float(np.nanmin(synthetic_samples)), float(np.nanmax(synthetic_samples))]
    }
    torch.save(metadata, output_dir / f"{model_name}_metadata.pt")
    print(f"    ✓ Saved: {model_name}_metadata.pt")


def run_inference_all(models_config, output_dir='./synthetic_samples_fixed', S=100, device='cuda', test_subset=None):
    """
    Run inference on all models.
    
    Args:
        models_config: Dictionary mapping model names to (config_path, checkpoint_path, conditioning_type)
        output_dir: Directory to save outputs
        S: Number of samples per condition
        device: Device to run on
        test_subset: If not None, only generate for first N test conditions (for quick testing)
    
    """
    if test_subset is not None:
        print(f"TEST MODE: Using only first {test_subset} conditions")
    
    print("\nLoading normalization parameters...")
    target_mean = np.load('target_mean.npy')
    target_std = np.load('target_std.npy')
    
    print(f"  Target normalization:")
    print(f"    Shape: {target_mean.shape}")
    print(f"    Mean (first 5): {target_mean.flatten()[:5]}")
    print(f"    Std (first 5): {target_std.flatten()[:5]}")
    
    for model_name, (config_path, ckpt_path, conditioning_type) in models_config.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {ckpt_path}")
        if conditioning_type:
            print(f"  Conditioning type: {conditioning_type}")
        
        try:
            model, config = load_model(config_path, ckpt_path, device)
            
            is_unconditional = config['model'].get('is_unconditional', False)
            
            if is_unconditional:
                if test_subset is not None:
                    K_unconditional = test_subset
                else:
                    K_unconditional = 2144
                
                print(f"\n  Unconditional mode:")
                print(f"    Generating {K_unconditional} pseudo-conditions x {S} samples")
                
                synthetic_samples = generate_unconditional_samples(
                    model, K=K_unconditional, S=S, device=device
                )
                
                synth_mean = np.nanmean(synthetic_samples)
                synth_std = np.nanstd(synthetic_samples)
                print(f"\n  Generated data check:")
                print(f"    Mean: {synth_mean:.4f} (should be ~0)")
                print(f"    Std: {synth_std:.4f} (should be ~1)")
                
                if abs(synth_mean) > 1.0 or abs(synth_std - 1.0) > 0.5:
                    print(f"    ⚠️  WARNING: Data may not be properly normalized!")
                
                save_results(
                    output_dir, 
                    model_name, 
                    synthetic_samples,
                    target_mean=target_mean,
                    target_std=target_std
                )
            else:
                print(f"\n  Conditional mode:")
                print(f"    Loading test data...")
                
                loaders = get_dataloader(config, batch_size=32, shuffle=False)
                test_loader = loaders['test']
                K = len(test_loader.dataset)
                
                if test_subset is not None:
                    print(f"    K={K} total conditions, using first {test_subset} for testing")
                else:
                    print(f"    K={K} test conditions, S={S} samples each")
                
                synthetic_samples, real_samples, conditioning = generate_conditional_samples(
                    model, test_loader, S=S, device=device, max_conditions=test_subset
                )
                
                synth_mean = np.nanmean(synthetic_samples)
                synth_std = np.nanstd(synthetic_samples)
                real_mean = np.nanmean(real_samples)
                real_std = np.nanstd(real_samples)
                
                print(f"\n  Generated data check:")
                print(f"    Synthetic - Mean: {synth_mean:.4f}, Std: {synth_std:.4f}")
                print(f"    Real - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
                print(f"    (Both should have mean~0, std~1)")
                
                if abs(synth_mean) > 1.0 or abs(synth_std - 1.0) > 0.5:
                    print(f"    ⚠️  WARNING: Synthetic data may not be properly normalized!")
                if abs(real_mean) > 1.0 or abs(real_std - 1.0) > 0.5:
                    print(f"    ⚠️  WARNING: Real data may not be properly normalized!")
                
                save_results(
                    output_dir,
                    model_name,
                    synthetic_samples,
                    real_samples=real_samples,
                    conditioning=conditioning,
                    target_mean=target_mean,
                    target_std=target_std
                )
            
            print(f"\n  ✓ Successfully processed {model_name}")
            
        except Exception as e:
            print(f"\n  ✗ ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")
    print("\nIMPORTANT:")
    print("  - All samples are saved in NORMALIZED form (mean~0, std~1)")
    print("  - Use the saved normalization parameters to unnormalize when needed")
    print("  - Unnormalize formula: x_unnorm = x_norm * std + mean")


def main():
    """Main execution function."""
    
    # Configuration for all models
    models_config = {
        'combined': (
            './config/base_conditional_combined.yaml',
            './wandb/run-20260121_221702-pkljy2cy/files/diffusion-combined_20260121_221703/model_best_val.pth',
            'combined'
        ),
        'vae_only': (
            './config/base_conditional_vae.yaml',
            './wandb/run-20260121_221650-iy6aktzs/files/diffusion-vae-only_20260121_221651/model_best_val.pth',
            'vae_only'
        ),
        '2d_traits': (
            './config/base_conditional_2d.yaml',
            './wandb/run-20260121_221636-cmvpgui2/files/diffusion-2d-traits_20260121_221637/model_best_val.pth',
            '2d_only'
        ),
        'unconditional': (
            './config/base_unconditional.yaml',
            './wandb/run-20260121_221437-mb4zk0js/files/unconditional-diffusion_20260121_221438/model_best_val.pth',
            None
        )
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_subset = None 
    
    if test_subset is not None:
        print(f"\n⚠️  RUNNING IN TEST MODE: Only {test_subset} conditions")
        print(f"⚠️  Change test_subset=None in main() for full run\n")

    run_inference_all(
        models_config=models_config,
        output_dir='./synthetic_samples',
        S=100,
        device=device,
        test_subset=test_subset 
    )


if __name__ == "__main__":
    main()