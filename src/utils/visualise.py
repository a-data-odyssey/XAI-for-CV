from skimage import feature, color
import torch
import numpy as np

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

def grads_from_tensor(grads_in):
    """Conver tensor to numpy array"""

    grads = grads_in.squeeze().detach().cpu().numpy()
    grads = np.transpose(grads, (1, 2, 0))  # CHW → HWC

    return grads

def process_grads(grads_in,
                  activation="None",
                  skew=True,
                  normalize=True,
                  greyscale=False,
                  edge=None):
    """
    Process the gradients for visualization.

    Parameters:
        grads_in (np.array): Gradients to be processed.
        activation (str): Activation function to be applied to the gradients. Options: "relu", "abs".
        skew (bool): Whether to skew the gradients.
        normalize (bool): Whether to normalize the gradients.
        greyscale (bool): Whether to convert the gradients to greyscale.
        edge (np.array): Edge map to overlay on the gradients.

    Returns:
        np.array: Processed gradients.
    """

    #Convert tensor:
    if type(grads_in) == torch.Tensor:
        grads = grads_from_tensor(grads_in)
    else:
        # Copy the gradients
        grads = np.copy(grads_in)

        # Transpose the gradients
        if (len(grads.shape) >= 3) and (grads.shape[0] == 3):
            grads = np.transpose(grads, (1, 2, 0))

    # Get the absolute value of the gradients
    if activation == "relu":
        grads = np.maximum(0, grads)
    elif activation == "abs":
        grads = np.abs(grads)
    else:
        grads = grads

    # Normalize the gradients
    if normalize:
        grads -= np.min(grads)
        grads /= (np.max(grads)+1e-9)

    # Skew the gradients
    if skew:
        grads = np.sqrt(grads)

    # Convert the gradients to greyscale
    if greyscale:
        grads = np.mean(grads, axis=-1)

    if edge is not None:
        if greyscale == False:
            edge = np.expand_dims(edge, axis=2)  # make HWC
        # make 1 where edge is, original value elsewhere
        grads = np.minimum(grads, 1-edge)


    return grads

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
