import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, config, wandb_run, device):
        super().__init__()
        self.device = device
        self.target_dim = config["model"]['target_dim']
        self.horizon = config["model"]['horizon']
        self.is_unconditional = config["model"].get('is_unconditional', False)
        self.cond_dim = config["model"].get('cond_dim', 0)

        if wandb_run is not None:
            self.emb_time_dim = wandb_run.config.timeemb
            self.emb_feature_dim = wandb_run.config.featureemb
        else:
            self.emb_time_dim = config["wandb_run"]['config']['timeemb']
            self.emb_feature_dim = config["wandb_run"]['config']['featureemb']

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, 
            embedding_dim=self.emb_feature_dim
        )
        
        config_diff = config["diffusion"]
        
        if self.is_unconditional:
            config_diff["side_dim"] = self.emb_total_dim
        else:
            config_diff["side_dim"] = self.emb_total_dim + config["model"].get("cond_dim", 0)
        
        self.diffmodel = diff_CSDI(config_diff, inputdim=1)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, 
                config_diff["beta_end"] ** 0.5, 
                self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], 
                config_diff["beta_end"], 
                self.num_steps
            )
        else:
            raise ValueError(f"Unknown schedule: {config_diff['schedule']}")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        
        self.alpha_torch = (
            torch.tensor(self.alpha)
            .float()
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, conditioning_data):
        """Get side information for the diffusion model."""
        B, D, L = conditioning_data.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if not self.is_unconditional:
            B, D, T = conditioning_data.shape
            K = self.target_dim
            cond_info = conditioning_data.unsqueeze(2).expand(B, D, K, T)
            side_info = torch.cat([side_info, cond_info], dim=1)

        return side_info

    def calc_loss(self, observed_data, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        
        current_alpha = self.alpha_torch[t]

        noise = torch.randn_like(observed_data)
        noisy_data = (
            (current_alpha ** 0.5) * observed_data + 
            (1.0 - current_alpha) ** 0.5 * noise
        )

        total_input = noisy_data.unsqueeze(1)
        predicted = self.diffmodel(total_input, side_info, t)

        loss = F.mse_loss(predicted, noise)
        return loss

    def synthesize(self, batch=None, conditioning_data=None, batch_size=None):
        """
        Generate synthetic samples using DDPM reverse diffusion.
        
        Returns data in shape (B, K, L) where K=target_dim, L=horizon
        This is then converted to (B, L, K) = (B, T, D) in the generation script.
        """
        if batch is not None:
            conditioning_data = batch["conditioning_data"].to(self.device).float()
            B, D, L = conditioning_data.shape
            timepoints = batch["timepoints"].to(self.device).float()
        elif conditioning_data is not None:
            if isinstance(conditioning_data, torch.Tensor):
                B, D, L = conditioning_data.shape
                conditioning_data = conditioning_data.to(self.device)
                timepoints = torch.arange(L).to(self.device).unsqueeze(0).repeat(B, 1).float()
            else:
                raise ValueError(f"conditioning_data must be a tensor, got {type(conditioning_data)}")
        elif batch_size is not None:
            B = batch_size
            D = self.cond_dim if hasattr(self, 'cond_dim') and self.cond_dim > 0 else 1
            L = self.horizon
            conditioning_data = torch.zeros(B, D, L).to(self.device)
            timepoints = torch.arange(L).to(self.device).unsqueeze(0).repeat(B, 1).float()
        else:
            B = 32
            D = self.cond_dim if hasattr(self, 'cond_dim') and self.cond_dim > 0 else 1
            L = self.horizon
            conditioning_data = torch.zeros(B, D, L).to(self.device)
            timepoints = torch.arange(L).to(self.device).unsqueeze(0).repeat(B, 1).float()

        side_info = self.get_side_info(timepoints, conditioning_data)

        K = self.target_dim
        sample_output = torch.zeros(B, K, L).to(self.device)
        synthesized_output = torch.randn_like(sample_output)

        for tidx in reversed(range(self.num_steps)):
            input_data = synthesized_output.unsqueeze(1)  # (B, 1, K, L)
            
            predicted_noise = self.diffmodel(
                input_data, 
                side_info, 
                torch.tensor([tidx]).to(self.device)
            )

            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5
            
            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)

            if tidx > 0:
                noise = torch.randn_like(sample_output)
                sigma = (
                    (1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]
                ) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        
        if self.is_unconditional:
            B, K, L = observed_data.shape
            conditioning_data = torch.zeros(B, 0, L).to(self.device).float()
        else:
            conditioning_data = batch["conditioning_data"].to(self.device).float()

        side_info = self.get_side_info(observed_tp, conditioning_data)
        
        return self.calc_loss(observed_data, side_info, is_train)


class CSDI_PM25(CSDI_base):
    def __init__(self, config, wandb_run=None, device='cpu'):
        super(CSDI_PM25, self).__init__(config, wandb_run, device=device)