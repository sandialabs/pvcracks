# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 21:19:19 2025

@authors: nrjost
"""

def segment(
    image,
    device,
    model,
    category_mapping={0: "dark", 1: "busbar", 2: "crack", 3: "non-cell"},
    threshold=0.5,
):
    """
    Run segmentation model on an image to extract defect masks:
      - dark area mask
      - busbars mask
      - cracks mask
      - no-cell area mask

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image.Image
        Input image to segment.
    device : torch.device
        Computation device for running the model (e.g., CPU or CUDA).
    model : torch.nn.Module
        Pretrained segmentation network returning per-class logits.
    category_mapping : dict, optional
        Mapping from class index to display name.
        Default: {0: 'dark', 1: 'busbar', 2: 'crack', 3: 'non-cell'}.
    threshold : float, optional
        Probability cutoff for converting logits to binary masks.
        Default: 0.5.

    Returns
    -------
    dark, bb, crack, nocell : numpy.ndarray
        Binary masks for dark areas, busbars, cracks, and no-cell areas, respectively.
    """
    from torchvision import transforms
    import torchvision.transforms.functional as F
    import torch

    # Preprocess input
    img = transforms.ToTensor()(image).to(device)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Inference
    logits = model(img.unsqueeze(0)).detach().cpu()      # [1, n_classes, H, W]
    probs = torch.sigmoid(logits)                        # [1, n_classes, H, W]
    pred_mask = (probs > threshold).float().squeeze(0).numpy()  # [n_classes, H, W]

    dark = pred_mask[0]
    bb = pred_mask[1]
    crack = pred_mask[2]
    nocell = pred_mask[3]

    return dark, bb, crack, nocell