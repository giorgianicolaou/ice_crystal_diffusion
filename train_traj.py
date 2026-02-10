import argparse
import torch
import datetime
import json
import yaml
import os
from main_model import CSDI_PM25
from utils import train
import wandb

def get_dataloader_module(dataset_type):
    dataset_map = {
        'combined': 'dataset_crystaltraj',
        'vae_only': 'dataset_crystaltraj',
        '2d_only': 'dataset_crystaltraj',
        'unconditional': 'dataset_crystaltraj'
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. "
                        f"Must be one of {list(dataset_map.keys())}")
    
    module_name = dataset_map[dataset_type]
    
    try:
        module = __import__(module_name)
        return module.get_dataloader
    except ImportError as e:
        raise ImportError(f"Could not import {module_name}: {e}")


def validate_config(config, dataset_type):
    model_config = config["model"]
    
    if dataset_type == 'unconditional':
        model_config["cond_dim"] = 0
        model_config["is_unconditional"] = True
        model_config["conditioning_type"] = None
    elif dataset_type == 'combined':
        model_config["cond_dim"] = 64
        model_config["is_unconditional"] = False
        model_config["conditioning_type"] = 'combined'
    elif dataset_type == 'vae_only':
        model_config["cond_dim"] = 50
        model_config["is_unconditional"] = False
        model_config["conditioning_type"] = 'vae_only'
    elif dataset_type == '2d_only':
        model_config["cond_dim"] = 14
        model_config["is_unconditional"] = False
        model_config["conditioning_type"] = '2d_only'
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    return config


def train_model():
    parser = argparse.ArgumentParser(description="Trajectory Generation with Unified Model")
    parser.add_argument("--config", type=str, default="base.yaml",
                       help="Config file name (in config/ directory)")
    
    parser.add_argument("--dataset", type=str, default=None,
                       choices=['combined', 'vae_only', '2d_only', 'unconditional'],
                       help="Override dataset type from config")
    
    parser.add_argument("--model", type=str, help="Model architecture name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--timeemb", type=int, help="Time embedding dimension")
    parser.add_argument("--featureemb", type=int, help="Feature embedding dimension")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    args = parser.parse_args()

    config_path = f"config/{args.config}"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args.dataset is not None:
        dataset_type = args.dataset
    else:
        if config["model"].get("is_unconditional", False):
            dataset_type = 'unconditional'
        else:
            dataset_type = config["model"].get("conditioning_type", "combined")
    
    config = validate_config(config, dataset_type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelfolder = config["model"].get("model_folder", "")
    modeldirectory = config["model"].get("model_directory", "./saved_models")

    if 'wandb_run' in config:
        wandb_cfg = config["wandb_run"]
        
        wandb_config = wandb_cfg.get("config", {})
        wandb_config["dataset_type"] = dataset_type
        wandb_config["is_unconditional"] = config["model"]["is_unconditional"]
        wandb_config["conditioning_type"] = config["model"].get("conditioning_type")
        wandb_config["cond_dim"] = config["model"]["cond_dim"]
        
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "trajectory-generation"),
            name=wandb_cfg.get("run_name", None),
            config=wandb_config,
            tags=[dataset_type, "unified-model"]
        )
    else:
        wandb_run = wandb.init(
            project="trajectory-generation",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "model": args.model,
                "timeemb": args.timeemb,
                "featureemb": args.featureemb,
                "dataset_type": dataset_type,
                "is_unconditional": config["model"]["is_unconditional"],
                "conditioning_type": config["model"].get("conditioning_type"),
                "cond_dim": config["model"]["cond_dim"]
            },
            tags=[dataset_type, "unified-model"]
        )

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"{modeldirectory}/{wandb_run.name}_{current_time}/"
    os.makedirs(foldername, exist_ok=True)
    
    config_to_save = config.copy()
    config_to_save["dataset_type"] = dataset_type
    config_to_save["training_timestamp"] = current_time
    config_to_save["wandb_run_id"] = wandb_run.id
    config_to_save["wandb_run_name"] = wandb_run.name
    config_to_save["model_type"] = "unified"
    
    with open(foldername + "config.json", "w") as f:
        json.dump(config_to_save, f, indent=4)
    
    with open(foldername + "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    get_dataloader = get_dataloader_module(dataset_type)
    loaders = get_dataloader(config, wandb_run)
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    model = CSDI_PM25(config, wandb_run, device=device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb_run.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
    }, allow_val_change=True)

    if modelfolder == "":
        training_results = train(
            model, 
            config, 
            train_loader, 
            wandb_run, 
            device, 
            foldername=foldername, 
            valid_loader=val_loader
        )
    else:
        model_path = f"{modeldirectory}/{modelfolder}/model_best_val.pth"
        
        if not os.path.exists(model_path):
            model_path = f"{modeldirectory}/{modelfolder}/model.pth"
        
        model.load_state_dict(torch.load(model_path, map_location=device))

    if test_loader is not None:
        model.eval()
        test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                loss = model(batch)
                if not torch.isnan(loss):
                    test_loss += loss.item()
                    num_batches += 1
        
        if num_batches > 0:
            test_loss /= num_batches
            wandb_run.log({"test/loss": test_loss})
            
            test_results = {
                "test_loss": test_loss,
                "num_batches": num_batches,
                "dataset_type": dataset_type
            }
            with open(foldername + "test_results.json", "w") as f:
                json.dump(test_results, f, indent=4)
    
    wandb_run.finish()


if __name__ == "__main__":
    train_model()