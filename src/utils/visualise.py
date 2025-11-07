import numpy as np

from skimage import feature, color
import matplotlib.cm as cm

import torch

import cv2
from PIL import Image

import warnings
import urllib

def get_edge(image,h1=0.1,h2=0.3, sigma=2, as_image=False):
    """Get edge map from image"""

    gray_image = color.rgb2gray(image)
    edge = feature.canny(gray_image, sigma=sigma, low_threshold=h1, high_threshold=h2)

    # Format edge map as image
    if as_image:
        edge = 1 - np.float32(edge)
        edge = np.expand_dims(edge, axis=2)  # make HWC

    return edge

def add_edge_to_attributions(attributions, edge,edge_colour='black'):
    """ Add edge to attributions"""

    #Check to see if is numpy array
    assert isinstance(attributions, np.ndarray), "Attributions must be a numpy array"
    assert isinstance(edge, np.ndarray), "Edge must be a numpy array"
    attr_with_edge = attributions.copy()

    if edge_colour=='black':
        edge_value = np.min(attributions)
    elif edge_colour=='white':
        edge_value = np.max(attributions)

    print("edge value:",edge_value )

    # Overlay edge on attributions
    if attributions.ndim == 3 and edge.ndim == 2:
        #3 channel attributions and single channel edge
        for c in range(attributions.shape[2]):
            attr_with_edge[:,:,c][edge==True] = edge_value
    elif attributions.ndim == 2 and edge.ndim == 2:
        #Single channel attributions and single channel edge
        attr_with_edge[edge==True] = edge_value

    return attr_with_edge

def grads_from_tensor(grads_in):
    """Conver tensor to numpy array"""

    grads = grads_in.squeeze().detach().cpu().numpy()
    grads = np.transpose(grads, (1, 2, 0))  # CHW → HWC

    return grads

def process_attributions(
    raw_attributions: torch.Tensor,
    activation: str | None = None,  # ['abs', 'relu', 'none']
    normalize: bool = True,
    skew: float = 1.0,
    grayscale: bool = False,
    colormap: str | None = None,  # <-- Default: None (no colormap)
    output_as_pil: bool = False
):
    """
    Visualize gradient or attribution maps for image classification.

    Args:
        attribution (torch.Tensor): Attribution map with shape (C, H, W) or (H, W).
        activation (str): Activation applied to attributions ('abs', 'relu', or 'none').
        normalize (bool): Whether to normalize attributions to [0, 1].
        skew (float): Exponent to skew intensities (gamma correction). Only valid if normalize=True.
        grayscale (bool): If True, converts output to grayscale before applying colormap.
        colormap (str or None): Matplotlib colormap ('hot', 'coolwarm', etc.) or None for no color mapping.
        output_as_pil (bool): If True, returns a PIL Image instead of NumPy array.

    Returns:
        np.ndarray or PIL.Image: Visualized attribution map (RGB or grayscale).
    """

    # --- Safety checks ---
    if not normalize and skew != 1.0:
        raise ValueError("Cannot apply skew without normalization. Set normalize=True or skew=1.0.")
    if activation not in ["abs", "relu", None]:
        raise ValueError("Invalid activation. Choose from ['abs', 'relu', None].")
    if grayscale and colormap is not None:
        warnings.warn("Grayscale=True overrides colormap choice. Colormap will be ignored.", UserWarning)

    # Convert to NumPy
    if isinstance(raw_attributions, torch.Tensor):
        attribution = raw_attributions.detach().cpu().numpy()
    else:
        attribution = raw_attributions.copy()

    # Handle multi-channel case (1, C, H, W)
    if attribution.shape[0] == 1:
        attribution = attribution[0]
    if attribution.ndim == 3 and attribution.shape[0] == 3:
        attribution = np.transpose(attribution, (1, 2, 0))  # C, H, W -> H, W, C

    # --- Apply activation ---
    if activation == "abs":
        attribution = np.abs(attribution)
    elif activation == "relu":
        attribution = np.maximum(0, attribution)
    # 'none' → do nothing

    # --- Normalization ---
    if normalize:
        attribution -= attribution.min()
        if attribution.max() > 0:
            attribution /= attribution.max()

    # --- Skewing (gamma correction) ---
    if skew != 1.0:
        attribution = np.power(attribution, skew)

    # --- Visualization logic ---
    if grayscale or (colormap is not None):
        # Return single-channel map (grayscale look)
        if attribution.ndim == 3:
            attribution = np.mean(attribution, axis=-1)

        attribution_img = np.uint8(255 * np.clip(attribution, 0, 1))
        result = cv2.cvtColor(attribution_img, cv2.COLOR_GRAY2RGB)

    if colormap is not None:
        cmap = cm.get_cmap(colormap)
        colored = cmap(attribution)[:, :, :3]
        result = np.uint8(255 * colored)
    else:
        # No colormap, return as RGB
        result = attribution

    if output_as_pil:
        return Image.fromarray(result)
    return result

def display_imagenet_output(output,n=5):

    """Display the top n categories predicted by the model."""

    # Download the categories
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, "imagenet_classes.txt")

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top categories per image
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, n)

    for i in range(top_prob.size(0)):
        print(categories[top_catid[i]], top_prob[i].item())

    return top_catid[0]
