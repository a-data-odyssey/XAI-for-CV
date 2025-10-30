import torch
import numpy as np
from torchvision.io import read_image

from PIL import Image

from torchvision import transforms

def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

def preprocess_imagenet_image(img_path):

    """Load and preprocess images for PyTorch models."""

    img = Image.open(img_path).convert("RGB")

    #Transforms used by imagenet models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(img).unsqueeze(0)

# Classes
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        """Initialise the dataset
        Args:
        paths (list): List of file paths
        target_pos (int): Position of the target band in the input data
        incl_bands (list): List of bands to include in the input data
        satellite (str): Satellite type"""

        self.paths = paths

        # Define dataset parameters
        self.target = 9
        self.incl_bands = [0, 1, 2, 3, 4, 5, 6]
        self.satellite = "landsat"

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        bands = instance[:, :, self.incl_bands]  # Only include specified bands
        bands = bands.astype(np.float32)

        # Normalise bands
        bands = scale_bands(bands, self.satellite)

        # Convert to tensor
        bands = bands.transpose(2, 0, 1)
        bands = torch.tensor(bands, dtype=torch.float32)

        # Get target
        mask_1 = instance[:, :, self.target].astype(np.int8)  # Water = 1, Land = 0
        mask_1[np.where(mask_1 == -1)] = 0  # Set nodata values to 0
        mask_0 = 1 - mask_1

        target = np.array([mask_0, mask_1])
        target = torch.Tensor(target).squeeze()

        return bands, target

    def __len__(self):
        return len(self.paths)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, num_classes):
        self.paths = paths
        self.num_classes = num_classes  # Number of classes for classification

    def __getitem__(self, idx):
        """Get image and target"""
        # Read image
        path = self.paths[idx]
        image = read_image(path).float() / 255.0  # Normalize image

        # Get target based on task type
        if self.num_classes == 1:
            # For regression, extract target value from filename
            target = float(path.split("/")[-1].split("_")[0])  # Assuming regression target is encoded in the filename
            target = torch.Tensor([target])
        else:
            # Extract class label from filename
            target_label = int(path.split("/")[-1].split("_")[0])  # Assuming class is encoded in filename
            # One-hot encode the target
            target = torch.zeros(self.num_classes)
            target[target_label] = 1

        return image, target

    def __len__(self):
        return len(self.paths)
