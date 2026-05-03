import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

print("CWD:", os.getcwd())
print("Files:", os.listdir())

from VAE import VAELightningModule
from Data_loader import CrystalDataModule
from config import train
import pandas as pd

crystals_filename = '/glade/derecho/scratch/joko/cpi/hysplit/CRYSTALS.parquet'
crystal_df = pd.read_parquet(crystals_filename)
crystal_df = crystal_df.dropna(subset=["WRF_LAT"])
crystal_df.index.name = 'crystal'

image_base_path = '/glade/derecho/scratch/joko/cpi/CRYSTAL_FACE_UND'
crystal_df['image_path'] = crystal_df['filename'].apply(lambda f: os.path.join(image_base_path, f))
crystal_df = crystal_df[crystal_df['image_path'].apply(os.path.exists)]

path_model = train['model_directory']
os.makedirs(path_model, exist_ok=True)
ckpt_path = "/glade/u/home/gnicolaou/ice-summer-2025/VAE/model/best_model-epoch=48-val_loss=0.00.ckpt"

logger = TensorBoardLogger("tb_logs", name="crystal_vae")
torch.set_float32_matmul_precision('medium')
strategy = DeepSpeedStrategy()
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(path_model + "/tb_logs/profiler0"),
    schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="Validation_epoch_loss",
    mode="min",
    dirpath=path_model,
    filename='best_model-{epoch:02d}-{val_loss:.2f}'
)

model = VAELightningModule(
    lr=train['learning_rate'],
    latent_dim=train.get('latent_dim', 50)
)

dm = CrystalDataModule(
    dataframe=crystal_df,
    batch_size=train['batch_size'],
    num_workers=train['number_of_workers']
)

world_size = torch.cuda.device_count()

trainer = pl.Trainer(
    accelerator="gpu",
    devices=list(range(world_size)),
    max_epochs=train['epochs'],
    resume_from_checkpoint=ckpt_path,
    profiler=profiler,
    strategy=strategy,
    log_every_n_steps=100,
    default_root_dir=path_model,
    callbacks=[checkpoint_callback],
    logger=logger
)

trainer.fit(model, dm)