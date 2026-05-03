import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import numpy as np

class CrystalImageDataset(Dataset):
    def __init__(self, dataframe, target_size=(128, 128)):
        self.df = dataframe.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['image_path']
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_size, Image.BILINEAR)
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1)  # [C, H, W]
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            img = torch.zeros(3, *self.target_size)  # fallback image

        return img

class CrystalDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, batch_size=32, num_workers=4):
        super().__init__()
        self.df = dataframe
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = CrystalImageDataset(self.df)
        total = len(dataset)
        val_size = total // 10
        test_size = total // 10
        train_size = total - val_size - test_size
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

