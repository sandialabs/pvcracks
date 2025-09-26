# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from utils.unet_model import construct_unet

# %%
# root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_ASU/"
# root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_CWRU_Dupont/"
# root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_CWRU_SunEdison/"
# root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_LBNL/"
root = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU/"


model_weight_paths = {
    "emma_retrained": "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/retrained_pv-vision_model.pt",
    "original": "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/pv-vision_model.pt",
}

# weight_path = model_weight_paths["emma_retrained"]
weight_path = model_weight_paths["original"]

checkpoint_name = "wandb_" + root.split("/")[-2]


# %%
def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
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

    train_dataset = functions.SolarDataset(
        root, image_folder="img/train", mask_folder="ann/train", transforms=transformers
    )

    val_dataset = functions.SolarDataset(
        root, image_folder="img/val", mask_folder="ann/val", transforms=transformers
    )

    return train_dataset, val_dataset


# %%
def load_device_and_model(weight_path):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    unet = construct_unet(5)
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
train_dataset, val_dataset = load_dataset(root)
device, model = load_device_and_model(weight_path)

# %%
category_mapping = {0: "empty", 1: "dark", 2: "busbar", 3: "crack", 4: "non-cell"}


# %%
def new_inference_and_show(idx, threshold=0.5):
    # Get the preprocessed image and multi-hot ground truth mask
    img, mask = train_loader.dataset.__getitem__(idx)
    img = img.to(device)

    # Get the raw image for display (assuming __getraw__ returns a PIL image)
    raw_img, _ = train_loader.dataset.__getraw__(idx)

    # --- Run inference ---
    # Get raw logits from the model, then apply Sigmoid and threshold
    logits = model(img.unsqueeze(0)).detach().cpu()  # shape: [1, 5, H, W]
    probs = torch.sigmoid(logits)  # shape: [1, 5, H, W]
    pred_mask = (probs > threshold).float().squeeze(0).numpy()  # shape: [5, H, W]

    # Ground truth is assumed to be already a 5-channel multi-hot mask.
    gt_mask = mask.cpu().numpy()  # shape: [5, H, W]

    # --- Visualization ---
    # Create a grid with 3 rows and 5 columns:
    #   Row 0: Raw image (displayed only once in the first column)
    #   Row 1: Ground truth masks for each class
    #   Row 2: Predicted masks for each class
    n_classes = 5
    class_names = [f"({k}) {v}" for k, v in category_mapping.items()]

    fig, axs = plt.subplots(3, n_classes, figsize=(4 * n_classes, 12))

    # Row 0: Display raw image in first subplot; hide other subplots in this row.
    axs[0, 0].imshow(raw_img.convert("L"), cmap="viridis")
    axs[0, 0].set_title("Raw Image")
    axs[0, 0].axis("off")
    for j in range(1, n_classes):
        axs[0, j].axis("off")

    # Row 1: Ground truth for each class (each channel)
    for j in range(n_classes):
        axs[1, j].imshow(gt_mask[j], cmap="viridis")
        axs[1, j].set_title(f"GT: {class_names[j]}")
        axs[1, j].axis("off")

    # Row 2: Predictions for each class (each channel)
    for j in range(n_classes):
        axs[2, j].imshow(pred_mask[j], cmap="viridis")
        axs[2, j].set_title(f"Pred: {class_names[j]}")
        axs[2, j].axis("off")

    fig.suptitle("Retrained Model Prediction", fontsize=16)

    plt.tight_layout()
    plt.show()


# %%
def create_wandb_image(idx, threshold=0.5):
    # Get the preprocessed image and multi-hot ground truth mask
    img, mask = train_loader.dataset.__getitem__(idx)
    img = img.to(device)

    # --- Run inference ---
    # Get raw logits from the model, then apply Sigmoid and threshold
    logits = model(img.unsqueeze(0)).detach().cpu()  # shape: [1, 5, H, W]
    probs = torch.sigmoid(logits)  # shape: [1, 5, H, W]
    pred_mask = (probs > threshold).float().squeeze(0).numpy()  # shape: [5, H, W]

    # Ground truth is assumed to be already a 5-channel multi-hot mask.
    gt_mask = mask.cpu().numpy()  # shape: [5, H, W]

    n_classes = len(category_mapping)

    this_id_mask_images = []
    for i in range(n_classes):
        masks_dict = {
            "predictions": {
                "mask_data": pred_mask[i],
                "class_labels": category_mapping,
            },
            "ground_truth": {
                "mask_data": gt_mask[i],
                "class_labels": category_mapping,
            },
        }

        mask_img = wandb.Image(
            img,
            masks=masks_dict,
        )
        this_id_mask_images.append(mask_img)
    return this_id_mask_images


# %%
def create_wandb_image_for_table(idx, threshold=0.5):
    # Get the preprocessed image and multi-hot ground truth mask
    img, mask = train_loader.dataset.__getitem__(idx)
    img = img.to(device)

    # --- Run inference ---
    # Get raw logits from the model, then apply Sigmoid and threshold
    logits = model(img.unsqueeze(0)).detach().cpu()  # shape: [1, 5, H, W]
    probs = torch.sigmoid(logits)  # shape: [1, 5, H, W]
    pred_mask = (probs > threshold).float().squeeze(0).numpy()  # shape: [5, H, W]

    # Ground truth is assumed to be already a 5-channel multi-hot mask.
    gt_mask = mask.cpu().numpy()  # shape: [5, H, W]

    n_classes = len(category_mapping)

    this_id_table_info = []
    this_id_table_info.append(wandb.Image(img))

    for i in range(n_classes):
        this_id_table_info.append(
            wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": gt_mask[i],
                        "class_labels": {0: category_mapping[i]},
                    },
                },
            )
        )
        this_id_table_info.append(
            wandb.Image(
                img,
                masks={
                    "predictions": {
                        "mask_data": pred_mask[i],
                        "class_labels": {0: category_mapping[i]},
                    },
                },
            )
        )

    return this_id_table_info


# %% [markdown]
# # Training

# %%
save_dir = get_save_dir(str(root), checkpoint_name)
os.makedirs(save_dir, exist_ok=True)

original_config = {
    "batch_size_train": 8,
    "lr": 0.00092234,
    "gamma": 0.11727,
    "num_epochs": 45,
    "batch_size_val": 8,
    "criterion": torch.nn.BCEWithLogitsLoss(),
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
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size_train, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False)

# %%
optimizer = Adam(model.parameters(), lr=config.lr)
# lr_scheduler = StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.gamma)

save_name = "model.pt"

# %%
# log gradients
run.watch(model, log_freq=100)

# %%
training_epoch_loss = []
val_epoch_loss = []

for epoch in tqdm(range(1, config.num_epochs + 1)):
    training_step_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.float()

        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # calc loss -- bce with logits loss applies sigmoid interally
        training_loss = original_config["criterion"](output, target)

        # backward pass
        training_loss.backward()
        optimizer.step()

        # record loss
        training_step_loss.append(training_loss.item())

    training_epoch_loss.append(np.array(training_step_loss).mean())

    val_step_loss = []

    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        target = target.float()

        # forward pass
        data = data.to(device)

        output = model(data)

        # calc loss -- bce with logits loss applies sigmoid interally
        val_loss = original_config["criterion"](output, target)

        val_step_loss.append(val_loss.item())

    val_epoch_loss.append(np.array(val_step_loss).mean())

    # Compute dice and IoU metrics per channel
    pred_probs = torch.sigmoid(output)
    pred_binary = (pred_probs > 0.5).float()

    dice_scores = []
    iou_scores = []

    for i in range(pred_binary.size(1)):  # Loop over channels
        dice = dice_coefficient(pred_binary[:, i], target[:, i])
        iou = iou_score(pred_binary[:, i], target[:, i])
        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(
        f"Epoch {epoch}/{config.num_epochs}, Training Loss: {np.array(training_step_loss).mean()}, Validation Loss: {np.array(val_step_loss).mean()}, Avg Dice: {avg_dice}, Avg IoU: {avg_iou}"
    )

    # print("Generating predictions for wandb...")
    # mask_images = []
    # table = wandb.Table(
    #     columns=[
    #         "Image",
    #         "GT Empty",
    #         "Pred Empty",
    #         "GT Dark",
    #         "Pred Dark",
    #         "GT Busbar",
    #         "Pred Busbar",
    #         "GT Crack",
    #         "Pred Crack",
    #         "GT Non-cell",
    #         "Pred Non-cell",
    #     ]
    # )
    # for id in range(20):
    #     mask_images.extend(create_wandb_image(id))
    #     new_img_table = create_wandb_image_for_table(id)
    #     table.add_data(*new_img_table)

    print("Logging to wandb...")
    run.log(
        {
            "train_loss": np.array(training_step_loss).mean(),
            "val_loss": np.array(val_step_loss).mean(),
            "avg_dice": avg_dice,
            "avg_iou": avg_iou,
            # "predictions": mask_images,
            # "table": table,
        }
    )

    table = wandb.Table(columns=["Image"])

    print("Saving model...")
    os.makedirs(os.path.join(save_dir, f"epoch_{epoch}"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}", save_name))
    print(f"Saved model at epoch {epoch}.", end=" ")

    if epoch >= 2 and epoch < config.num_epochs:
        os.remove(os.path.join(save_dir, f"epoch_{epoch - 1}", save_name))
        print(f"Removed model at epoch {epoch - 1}.", end="")
    print("\n")

# %% [markdown]
# ---

# %%
run.finish()
