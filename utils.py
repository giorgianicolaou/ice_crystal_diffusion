import numpy as np
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import wandb

def train( 
    model,
    config,
    train_loader,
    wandb_run,
    device,
    valid_loader=None,
    valid_epoch_interval=1,
    foldername="",  
):
    learning_rate = getattr(wandb_run.config, 'learning_rate', config['train']['learning_rate'])
    epochs = getattr(wandb_run.config, 'epochs', config['train']['epochs'])
    weight_decay = config['train'].get('weight_decay', 1e-6)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( 
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    plot_dir = os.path.join(foldername, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    
    train_losses = []
    val_losses = []
    
    for epoch_no in range(epochs):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                
                loss = model(train_batch)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss / batch_no,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch_no,
                })
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    },
                    refresh=False,
                )
        
        epoch_avg_loss = avg_loss / len(train_loader)
        train_losses.append(epoch_avg_loss)
        
        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for val_batch in valid_loader:
                    loss = model(val_batch)
                    
                    if torch.isnan(loss):
                        continue
                    
                    val_loss += loss.item()
            
            val_loss /= len(valid_loader)
            val_losses.append(val_loss)

            wandb_run.log({
                "val/loss": val_loss,
                "val/best_loss": best_val_loss,
                "epoch": epoch_no,
            })

            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_epoch = epoch_no
                
                best_model_path = os.path.join(foldername, "model_best_val.pth")
                torch.save(model.state_dict(), best_model_path)
                
                checkpoint_path = os.path.join(foldername, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch_no,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': epoch_avg_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                
                wandb_run.save(best_model_path)
        
        if (epoch_no + 1) % 10 == 0:
            checkpoint_path = os.path.join(foldername, f"checkpoint_epoch_{epoch_no}.pth")
            torch.save({
                'epoch': epoch_no,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': epoch_avg_loss,
                'val_loss': val_losses[-1] if val_losses else None,
            }, checkpoint_path)
    
    final_model_path = os.path.join(foldername, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    wandb_run.log({
        "train/final_loss": train_losses[-1],
        "val/final_loss": val_losses[-1] if val_losses else None,
        "train/best_val_loss": best_val_loss,
        "train/best_epoch": best_epoch,
    })
    
    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


# def quantile_loss(target, forecast, q: float, eval_points) -> float:
#     return 2 * torch.sum(
#         torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
#     )


# def calc_denominator(target, eval_points):
#     return torch.sum(torch.abs(target * eval_points))


# def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
#     target = target * scaler + mean_scaler
#     forecast = forecast * scaler + mean_scaler

#     quantiles = np.arange(0.05, 1.0, 0.05)
#     denom = calc_denominator(target, eval_points)
#     CRPS = 0
#     for i in range(len(quantiles)):
#         q_pred = []
#         for j in range(len(forecast)):
#             q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
#         q_pred = torch.cat(q_pred, 0)
#         q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
#         CRPS += q_loss / denom
#     return CRPS.item() / len(quantiles)


# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0
#         mae_total = 0
#         evalpoints_total = 0

#         all_target = []
#         all_observed_point = []
#         all_observed_time = []
#         all_evalpoint = []
#         all_generated_samples = []
#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 output = model.evaluate(test_batch, nsample)

#                 samples, c_target, eval_points, observed_points, observed_time = output
#                 samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
#                 c_target = c_target.permute(0, 2, 1)  # (B,L,K)
#                 eval_points = eval_points.permute(0, 2, 1)
#                 observed_points = observed_points.permute(0, 2, 1)

#                 samples_median = samples.median(dim=1)
#                 all_target.append(c_target)
#                 all_evalpoint.append(eval_points)
#                 all_observed_point.append(observed_points)
#                 all_observed_time.append(observed_time)
#                 all_generated_samples.append(samples)

#                 mse_current = (
#                     ((samples_median.values - c_target) * eval_points) ** 2
#                 ) * (scaler ** 2)
#                 mae_current = (
#                     torch.abs((samples_median.values - c_target) * eval_points) 
#                 ) * scaler

#                 mse_total += mse_current.sum().item()
#                 mae_total += mae_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()

#                 it.set_postfix(
#                     ordered_dict={
#                         "rmse_total": np.sqrt(mse_total / evalpoints_total),
#                         "mae_total": mae_total / evalpoints_total,
#                         "batch_no": batch_no,
#                     },
#                     refresh=True,
#                 )

#             with open(
#                 foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 all_target = torch.cat(all_target, dim=0)
#                 all_evalpoint = torch.cat(all_evalpoint, dim=0)
#                 all_observed_point = torch.cat(all_observed_point, dim=0)
#                 all_observed_time = torch.cat(all_observed_time, dim=0)
#                 all_generated_samples = torch.cat(all_generated_samples, dim=0)

#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         all_observed_time,
#                         scaler,
#                         mean_scaler,
#                     ],
#                     f,
#                 )

#             CRPS = calc_quantile_CRPS(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )

#             with open(
#                 foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 pickle.dump(
#                     [
#                         np.sqrt(mse_total / evalpoints_total),
#                         mae_total / evalpoints_total,
#                         CRPS,
#                     ],
#                     f,
#                 )
#                 print("RMSE:", np.sqrt(mse_total / evalpoints_total))
#                 print("MAE:", mae_total / evalpoints_total)
#                 print("CRPS:", CRPS)