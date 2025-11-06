import matplotlib.pyplot as plt
import torch


def channeled_inference_and_show(
    data_loader, device, model, category_mapping, idx, threshold=0.5
):
    """Run inference on a single dataset element and visualize predictions per class.

    Args:
        data_loader (torch.utils.data.DataLoader): Loader that provides dataset access.
        device (torch.device): Computation device for running the model.
        model (torch.nn.Module): Trained segmentation network returning channel logits.
        category_mapping (Mapping[int, str]): Mapping from class index to display name.
        idx (int): Index of the sample to visualize inside the dataset.
        threshold (float, optional): Probability cutoff for converting logits to masks.
    """
    # Get the preprocessed image and multi-hot ground truth mask
    img, mask = data_loader.dataset.__getitem__(idx)
    img = img.to(device)

    # Get the raw image for display (assuming __getraw__ returns a PIL image)
    raw_img, _ = data_loader.dataset.__getraw__(idx)

    # --- Run inference ---
    # Get raw logits from the model, then apply Sigmoid and threshold
    logits = model(img.unsqueeze(0)).detach().cpu()  # shape: [1, n_classes, H, W]
    probs = torch.sigmoid(logits)  # shape: [1, n_classes, H, W]
    pred_mask = (
        (probs > threshold).float().squeeze(0).numpy()
    )  # shape: [n_classes, H, W]

    # Ground truth is assumed to be already a n_classes-channel multi-hot mask.
    gt_mask = mask.cpu().numpy()  # shape: [n_classes, H, W]

    # --- Visualization ---
    # Create a grid with 3 rows and 4 columns:
    #   Row 0: Raw image (displayed only once in the first column)
    #   Row 1: Ground truth masks for each class
    #   Row 2: Predicted masks for each class
    n_classes = len(category_mapping)
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

    fig.suptitle("Model Prediction", fontsize=16)

    plt.tight_layout()
    plt.show()
