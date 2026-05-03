# Conditional Diffusion Models for Inferring Thermodynamic Histories from Ice Crystal Imagery

This repository contains the code accompanying the paper submitted to *Journal of Geophysical Research: Machine Learning and Computation*:

Nicolaou, G., Frields, K., Stephens, T., Cai, Y., Sulia, K., Przybylo, V., Watson-Parris, D., Ko, J., & Lamb, K. D. (2026). 
Inferring Thermodynamic Histories from In Situ Ice Crystal Imagery via Conditional Diffusion Models. 

## Overview

We introduce a machine learning framework for probabilistically inferring 
atmospheric thermodynamic histories from single-snapshot in situ ice crystal 
imagery. A conditional diffusion model, trained on variational autoencoder 
embeddings of CRYSTAL-FACE Cloud Particle Imager (CPI) images and crystal 
geometric attributes, generates ensemble distributions of temperature, 
pressure, and relative humidity trajectories consistent with observed 
crystal morphology.

The pipeline consists of:
1. A variational autoencoder (VAE) that learns compact morphological 
   embeddings from CPI ice crystal images.
2. A conditional score-based diffusion model (adapted from CSDI) that 
   generates seven-hour thermodynamic back-trajectories conditioned on 
   crystal embeddings and geometric attributes.

## Repository Structure

```
.
├── vae/                        # Variational autoencoder for morphological embeddings
│   ├── train.py            # VAE training script
│   ├── VAE.py            # VAE architecture (encoder/decoder)
│   └── ...
├── batch_scripts/              # PBS batch scripts for cluster runs
├── config/                     # Model and training configuration files
├── dataset_crystaltraj.py      # Dataset class for crystal-trajectory pairs
├── diff_models.py              # Diffusion model architecture (CSDI-based)
├── main_model.py               # Main model wrapper and training logic
├── train_traj.py               # Training script for the diffusion model
├── synthetic_sample_generator.py  # Inference / sample generation
├── final_compare_models.py     # Evaluation: CRPS, JSD, comparison plots
├── utils.py                    # Helper functions
└── *.npy                       # Normalization statistics (means/stds)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning
- NumPy, SciPy, pandas
- Matplotlib, seaborn
- Weights & Biases (`wandb`) for experiment tracking (not necessary)
- DeepSpeed (for multi-GPU VAE training)

## Reproducing Paper Results

To reproduce the results reported in the paper:

1. Download the dataset.
2. Train the VAE on CPI imagery (`vae/train.py`).
3. Generate VAE embeddings for all crystals in the dataset.
4. Train both conditional and unconditional diffusion models with the 
   provided configs (`train_traj.py`).
5. Generate trajectory ensembles for test crystals 
   (`synthetic_sample_generator.py`).
6. Run evaluation scripts to compute CRPS, JSD, and produce comparison 
   plots (`final_compare_models.py`).

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{nicolaou2026inferring,
  author = {Nicolaou, Giorgia and Frields, Katherine and Stephens, Troy 
            and Cai, Yichen and Sulia, Kara and Przybylo, Vanessa and 
            Watson-Parris, Duncan and Ko, Joseph and Lamb, Kara D.},
  title = {Inferring Thermodynamic Histories from In Situ Ice Crystal 
           Imagery via Conditional Diffusion Models},
  journal = {Journal of Geophysical Research: Machine Learning and 
             Computation},
  year = {2026}
}
```

## Acknowledgments

This work was supported by NSF through the Learning the Earth with 
Artificial Intelligence and Physics (LEAP) Science and Technology Center 
(Award #2019625) and by DOE under Grants DE-SC0021033 and DE-SC0023020.

The diffusion model architecture builds on CSDI (Tashiro et al., 2021).

## Contact

For questions about the code or paper, contact Giorgia Nicolaou 
(gnicolaou@ucsd.edu) or open an issue on this repository.
