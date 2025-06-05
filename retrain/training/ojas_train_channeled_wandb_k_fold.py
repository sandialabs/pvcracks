# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import wandb

# pv_vision_dir = os.path.join(Path.home(), 'pv-vision')
pv_vision_dir = os.path.join("/home/eccoope", "pv-vision")
# functions_dir = os.path.join(Path.home(), 'el_img_cracks_ec', 'scripts')
functions_dir = os.path.join("/home/eccoope", "el_img_cracks_ec", "scripts")

sys.path.append(pv_vision_dir)
sys.path.append(functions_dir)

# ojas_functions_dir = os.path.join(Path.home(), 'pvcracks/retrain/')
ojas_functions_dir = "/Users/ojas/Desktop/saj/SANDIA/pvcracks/retrain/"
sys.path.append(ojas_functions_dir)

import functions
from tutorials.unet_model import construct_unet


# %%
root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU_No_Empty/"
weight_path = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/pv-vision_model.pt"

checkpoint_name = "wandb_" + root.split("/")[-2]

# %%
category_mapping = {0: "dark", 1: "busbar", 2: "crack", 3: "non-cell"}

# %%
def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def iou_score(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou


# %%
def load_dataset(root):
    transformers = functions.Compose(
        [functions.ChanneledFixResize(256), functions.ToTensor(), functions.Normalize()]
    )

    full_dataset = functions.SolarDataset(
        root, image_folder="img/all", mask_folder="ann/all", transforms=transformers
    )

    return full_dataset

# %%
def load_device_and_model(weight_path):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    unet = construct_unet(len(category_mapping))
    unet = torch.nn.DataParallel(unet)

    model = unet.module.to(device)

    return device, model

# %%
def get_save_dir(base_dir, checkpoint_name):
    checkpoint_dir = base_dir + "/checkpoints/"
    folders = [folder for folder in os.listdir(checkpoint_dir)]

    max_number = 0
    for folder in folders:
        number = int(folder[-1])
        if number > max_number:
            max_number = number

    new_folder_name = f"{checkpoint_name}{max_number + 1}"
    new_folder_path = os.path.join(checkpoint_dir, new_folder_name)

    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path

# %%
full_dataset = load_dataset(root)

# %%
trainval_set, test_set = train_test_split(full_dataset, test_size=0.1, random_state=42)

# %% [markdown]
#  # Training

# %%
save_name = "model.pt"
save_dir = get_save_dir(str(root), checkpoint_name)
os.makedirs(save_dir, exist_ok=True)

original_config = {
    "batch_size_train": 8,
    "lr": 0.00092234,
    "gamma": 0.11727,
    "num_epochs": 1,
    
    # constants
    "batch_size_val": 8,
    "criterion": torch.nn.BCEWithLogitsLoss(),
    "k_folds": 5,
    # "lr_scheduler_step_size": 1,
}

config_serializable = original_config.copy()
config_serializable["criterion"] = str(config_serializable["criterion"])

with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_serializable, f, ensure_ascii=False, indent=4)

run = wandb.init(
    project="pvcracks",
    entity="ojas-sanghi-university-of-arizona",
    config=original_config,
)
config = wandb.config

# %%
k_folds = config.k_folds
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to collect per-fold best metrics
fold_val_losses = []
fold_dice_scores = []
fold_iou_scores = []


for fold, (train_ids, val_ids) in enumerate(kfold.split(trainval_set)):
    print(f"\n--- FOLD {fold+1}/{k_folds} ---")

    train_subsampler = torch.utils.data.Subset(trainval_set, train_ids)
    train_loader = DataLoader(train_subsampler, batch_size=config.batch_size_train, shuffle=True)
    val_subsampler = torch.utils.data.Subset(trainval_set, val_ids)
    val_loader = DataLoader(val_subsampler, batch_size=config.batch_size_val, shuffle=False)

    # Initialize a fresh model and optimizer
    device, model = load_device_and_model(weight_path)
    optimizer = Adam(model.parameters(), lr=config.lr)
    run.watch(model, log_freq=100)
    
    best_fold_val_loss = float("inf")
    best_fold_dice = 0.0
    best_fold_iou = 0.0
    
    # PER-EPOCH TRAINING
    for epoch in tqdm(range(1, config.num_epochs + 1)):
        training_step_loss = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()

            optimizer.zero_grad()
            output = model(data)
            training_loss = original_config["criterion"](output, target)
            training_loss.backward()
            optimizer.step()

            training_step_loss.append(training_loss.item())

        val_step_loss = []
        dice_scores = []
        iou_scores = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            output = model(data)
            
            val_loss = original_config["criterion"](output, target)
            val_step_loss.append(val_loss.item())

            # compute dice and iou
            pred_probs = torch.sigmoid(output)
            pred_binary = (pred_probs > 0.5).float()
            for i in range(pred_binary.size(1)):
                dice = dice_coefficient(pred_binary[:, i], target[:, i])
                iou = iou_score(pred_binary[:, i], target[:, i])
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

        epoch_train_loss = np.mean(training_step_loss)
        epoch_val_loss = np.mean(val_step_loss)
        epoch_avg_dice = np.mean(dice_scores)
        epoch_avg_iou = np.mean(iou_scores)

        # Log per-fold, per-epoch to W&B
        run.log({
            f"fold{fold+1}/train_loss": epoch_train_loss,
            f"fold{fold+1}/val_loss":   epoch_val_loss,
            f"fold{fold+1}/dice":       epoch_avg_dice,
            f"fold{fold+1}/iou":        epoch_avg_iou,
        }, step=epoch)

        # Keep best for this fold
        if epoch_val_loss < best_fold_val_loss:
            best_fold_val_loss = epoch_val_loss
            best_fold_dice = epoch_avg_dice
            best_fold_iou = epoch_avg_iou
    
    print(f"Fold {fold+1} best val_loss: {best_fold_val_loss:.4f}, dice: {best_fold_dice:.4f}, iou: {best_fold_iou:.4f}")
    
    fold_val_losses.append(best_fold_val_loss)
    fold_dice_scores.append(best_fold_dice)
    fold_iou_scores.append(best_fold_iou)

# %%
# ========== AGGREGATE RESULTS ACROSS FOLDS ==========

avg_val_loss = np.mean(fold_val_losses)
avg_dice     = np.mean(fold_dice_scores)
avg_iou      = np.mean(fold_iou_scores)

# Log the averages to W&B summary for sweep optimization
wandb.log({
    "avg_val_loss": avg_val_loss,
    "avg_dice":     avg_dice,
    "avg_iou":      avg_iou,
})
wandb.run.summary["avg_val_loss"] = avg_val_loss

print(f"Average val_loss: {avg_val_loss:.4f}, dice: {avg_dice:.4f}, iou: {avg_iou:.4f}")

# %% [markdown]
#  ---

# %%
run.finish()


# %% [markdown]
# ---


