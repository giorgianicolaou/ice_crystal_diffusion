import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
import os

NAN_PLACEHOLDER = 0

class TrajectoryDataset(Dataset):
    def __init__(self, external_data, target_data, is_unconditional=False):
        self.is_unconditional = is_unconditional
        self.external_data = external_data.permute(0, 2, 1).float()
        self.target_data = target_data.permute(0, 2, 1).float()
        self.length = self.external_data.shape[-1]
        self.feature_dim = self.external_data.shape[1]

    def __len__(self):
        return self.external_data.shape[0]

    def __getitem__(self, idx):
        sample = {
            "observed_data": self.target_data[idx],
            "timepoints": torch.arange(self.length).float()
        }
        if not self.is_unconditional:
            sample["conditioning_data"] = self.external_data[idx]
        return sample

def z_score_normalize(x, placeholder=NAN_PLACEHOLDER):
    """
    Z-score normalize data.
    Normalizes ALL values including placeholders.
    """
    mean = np.mean(x, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.std(x, axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    x_norm = (x - mean) / std
    
    print(f"  Normalization: all {x.size} values normalized")
    
    return x_norm, mean, std


def load_vae_embeddings(trimmed_crystal_df):
    print("  Loading VAE embeddings...")
    
    und_crystal_features = torch.load('/glade/u/home/gnicolaou/ice-summer-2025/VAE/trainfeat.pth')
    nasa_crystal_features = torch.load('/glade/u/home/gnicolaou/ice-summer-2025/VAE/trainfeat_und.pth')
    all_crystal_features = torch.cat([und_crystal_features, nasa_crystal_features], dim=0)
    all_crystal_features = np.asarray(all_crystal_features)
    
    with open('/glade/u/home/gnicolaou/ice-summer-2025/VAE/filelist.txt', 'r') as f:
        und_crystal_paths = [line.strip() for line in f.readlines()]
    
    with open('/glade/u/home/gnicolaou/ice-summer-2025/VAE/filelist_und.txt', 'r') as f:
        nasa_crystal_paths = [line.strip() for line in f.readlines()]
    
    all_crystal_paths = und_crystal_paths + nasa_crystal_paths
    emb_crystal_paths = np.array(all_crystal_paths)
    
    if len(emb_crystal_paths) != len(all_crystal_features):
        raise ValueError(f"Mismatch: {len(emb_crystal_paths)} paths but {len(all_crystal_features)} embeddings")
    
    emb_basenames = np.array([os.path.basename(path) for path in emb_crystal_paths])
    emb_dict = {basename: idx for idx, basename in enumerate(emb_basenames)}
    
    n_crystals = len(trimmed_crystal_df)
    embeddings_list = []
    valid_indices = []
    
    for idx, file in enumerate(trimmed_crystal_df['filename']):
        file_basename = os.path.basename(file).strip()
        
        if file_basename in emb_dict:
            emb_idx = emb_dict[file_basename]
            embedding = all_crystal_features[emb_idx]
            
            if embedding.shape != (50,):
                continue
            
            embeddings_list.append(embedding)
            valid_indices.append(idx)
    
    embeddings = np.array(embeddings_list, dtype=np.float32)
    valid_mask = np.zeros(n_crystals, dtype=bool)
    valid_mask[valid_indices] = True
    
    matched_count = len(valid_indices)
    match_rate = 100 * matched_count / n_crystals
    
    print(f"    Matched: {matched_count}/{n_crystals} ({match_rate:.1f}%)")
    
    if matched_count == 0:
        raise ValueError("No VAE embeddings matched!")
    
    if match_rate < 50:
        print(f"    WARNING: Only {match_rate:.1f}% of crystals have embeddings")
    
    return embeddings, valid_mask


def load_2d_traits(trimmed_crystal_df):
    print("  Loading 2D traits...")
    
    vars_to_use = [
        'Particle Width', 'Particle Height', 'Cutoff', 'Blur', 'Contours',
        'Edges', 'Circularity', 'Contrast', 'Solidity', 'Complexity',
        'Aspect Ratio', 'Area Ratio', 'Hull Area', 'Equivalent Diameter'
    ]
    
    traits_2d = trimmed_crystal_df[vars_to_use].to_numpy().astype(np.float32)
    print(f"    Shape: {traits_2d.shape}")
    
    return traits_2d


def get_dataloader(config, wandb_run=None, batch_size=None, shuffle=True):
    print("\n" + "="*80)
    print("INITIALIZING DATALOADER")
    print("="*80)
    
    target_vars = config['model']['target_vars']
    horizon = config["model"]['horizon'] 
    is_unconditional = config["model"]['is_unconditional']
    conditioning_type = config["model"].get('conditioning_type', 'combined')
    
    print(f"Config: {target_vars}, horizon={horizon}, unconditional={is_unconditional}, type={conditioning_type}")
    
    if batch_size is not None:
        batch_size = batch_size
    elif wandb_run is not None:
        batch_size = wandb_run.config.batch_size
    else:
        batch_size = config["wandb_run"]['config']['batch_size']
    print(f"Batch size: {batch_size}")
    
    print("\nLoading target data...")
    traj_df = pd.read_parquet('/glade/derecho/scratch/joko/cpi/hysplit/TRAJ.parquet')
    traj_ds = xr.Dataset.from_dataframe(traj_df)
    traj_ds.load()
    target = np.stack([traj_ds[var].values for var in target_vars], axis=-1)

    print(f"  Raw shape: {target.shape}, NaN%: {100 * np.isnan(target).sum() / target.size:.2f}%")

    target = target[:, :horizon, :]
    print(f"  Trimmed to horizon: {target.shape}")

    target = np.nan_to_num(target, nan=NAN_PLACEHOLDER)

    placeholder_count = np.sum(np.abs(target - NAN_PLACEHOLDER) < 1e-3)
    print(f"  Placeholders: {placeholder_count} ({100*placeholder_count/target.size:.2f}%)")

    print("\nCreating data splits (before filtering)...")
    indices = np.arange(target.shape[0])
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"  Initial splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    if not is_unconditional:
        print(f"\nLoading conditioning data (type: {conditioning_type})...")
        
        crystal_df = pd.read_parquet('/glade/derecho/scratch/joko/cpi/hysplit/CRYSTALS.parquet')
        trimmed_crystal_df = crystal_df.dropna(subset=['WRF_LAT'])
        trimmed_crystal_df.index.name = 'crystal'
        
        traj_crystals = traj_df.index.get_level_values("crystal").unique()
        trimmed_crystal_df = trimmed_crystal_df.loc[
            trimmed_crystal_df.index.intersection(traj_crystals)
        ].sort_index()
        
        print(f"  Crystals: {len(trimmed_crystal_df)}")
        
        print(f"\n  Loading VAE embeddings to determine valid samples...")
        vae_embeddings, vae_valid_mask = load_vae_embeddings(trimmed_crystal_df)
        
        print(f"  Filtering all data to crystals with VAE embeddings...")
        print(f"    Keeping {vae_valid_mask.sum()}/{len(vae_valid_mask)} crystals")
        
        trimmed_crystal_df = trimmed_crystal_df[vae_valid_mask]
        target = target[vae_valid_mask]
        
        old_to_new_idx = np.full(len(vae_valid_mask), -1, dtype=int)
        old_to_new_idx[vae_valid_mask] = np.arange(vae_valid_mask.sum())
        
        train_idx = old_to_new_idx[train_idx[vae_valid_mask[train_idx]]]
        val_idx = old_to_new_idx[val_idx[vae_valid_mask[val_idx]]]
        test_idx = old_to_new_idx[test_idx[vae_valid_mask[test_idx]]]
        
        train_idx = train_idx[train_idx >= 0]
        val_idx = val_idx[val_idx >= 0]
        test_idx = test_idx[test_idx >= 0]
        
        print(f"    After filtering - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        features_list = []
        feature_names = []
        vae_features = None
        traits_2d_features = None
        
        if conditioning_type in ['combined', 'vae_only']:
            vae_features = vae_embeddings
            features_list.append(vae_embeddings)
            feature_names.append('VAE')
        
        if conditioning_type in ['combined', '2d_only']:
            traits_2d_features = load_2d_traits(trimmed_crystal_df)
            features_list.append(traits_2d_features)
            feature_names.append('2D')
        
        if len(features_list) == 0:
            raise ValueError(f"No features loaded for conditioning_type={conditioning_type}")
        
        if vae_features is not None:
            print(f"\n  Normalizing VAE embeddings...")
            vae_norm, vae_mean, vae_std = z_score_normalize(
                vae_features[:, None, :]
            )
            vae_features = vae_norm.squeeze(1)
        
        if traits_2d_features is not None:
            print(f"\n  Normalizing 2D traits...")
            traits_2d_norm, traits_2d_mean, traits_2d_std = z_score_normalize(
                traits_2d_features[:, None, :]
            )
            traits_2d_features = traits_2d_norm.squeeze(1)
        
        features_list = []
        if vae_features is not None:
            features_list.append(vae_features)
            print(f"  VAE embeddings: shape {vae_features.shape} (normalized)")
        if traits_2d_features is not None:
            features_list.append(traits_2d_features)
            print(f"  2D traits: shape {traits_2d_features.shape} (normalized)")
        
        combined_features = np.concatenate(features_list, axis=1) if len(features_list) > 1 else features_list[0]
        print(f"  Combined features: {' + '.join(feature_names)}, shape: {combined_features.shape}")
        
        data = np.repeat(combined_features[:, None, :], horizon, axis=1)
        
        data = np.nan_to_num(data, nan=NAN_PLACEHOLDER)
        
        if conditioning_type == '2d_only':
            save_suffix = "_2d"
            np.save(f"data_mean{save_suffix}.npy", traits_2d_mean)
            np.save(f"data_std{save_suffix}.npy", traits_2d_std)
            print(f"  Saved data_mean{save_suffix}.npy and data_std{save_suffix}.npy")
        elif conditioning_type == 'vae_only':
            save_suffix = "_vae"
            np.save(f"data_mean{save_suffix}.npy", vae_mean)
            np.save(f"data_std{save_suffix}.npy", vae_std)
            print(f"  Saved data_mean{save_suffix}.npy and data_std{save_suffix}.npy")
        elif conditioning_type == 'combined':
            save_suffix = ""
            combined_mean = np.zeros((1, 1, combined_features.shape[1]), dtype=np.float32)
            combined_std = np.ones((1, 1, combined_features.shape[1]), dtype=np.float32)
            vae_dim = vae_features.shape[1]
            combined_mean[0, 0, :vae_dim] = vae_mean[0, 0, :]
            combined_std[0, 0, :vae_dim] = vae_std[0, 0, :]
            combined_mean[0, 0, vae_dim:] = traits_2d_mean[0, 0, :]
            combined_std[0, 0, vae_dim:] = traits_2d_std[0, 0, :]
            np.save(f"data_mean{save_suffix}.npy", combined_mean)
            np.save(f"data_std{save_suffix}.npy", combined_std)
            print(f"  Saved data_mean{save_suffix}.npy and data_std{save_suffix}.npy")
            print(f"    (Both VAE and 2D traits normalized)")
        else:
            raise ValueError(f"Unknown conditioning_type: {conditioning_type}")
        
    else:
        data = None
        print("\nUnconditional mode - no conditioning data")
    
    print("\nNormalizing target data...")
    target, target_mean, target_std = z_score_normalize(target)

    np.save("target_mean.npy", target_mean)
    np.save("target_std.npy", target_std)
    print(f"  Saved target_mean.npy and target_std.npy")
    print(f"  Final target shape: {target.shape}")
    
    print(f"\nFinal splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    target = torch.from_numpy(target)
    
    def create_loader(split_idx, split_name):
        y_split = target[split_idx]
        
        if not is_unconditional:
            x_split = data[split_idx]
            x_split = torch.from_numpy(x_split) if isinstance(x_split, np.ndarray) else x_split
        else:
            x_split = torch.empty((len(split_idx), horizon, 0))
        
        return DataLoader(
            TrajectoryDataset(x_split, y_split, is_unconditional=is_unconditional),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    print("\nCreating DataLoaders...")
    loaders = {
        'train': create_loader(train_idx, 'Train'),
        'val': create_loader(val_idx, 'Val'),
        'test': create_loader(test_idx, 'Test')
    }
    
    return loaders