import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_image(img: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Converts a PIL Image to a normalized PyTorch tensor ready for model input.

    Args:
        img (Image.Image): The input PIL image.
        target_size (Optional[Tuple[int, int]]): Optional (height, width) to resize the image.

    Returns:
        torch.Tensor: A 4D tensor of shape (1, 3, H, W) on CPU, normalized for ImageNet.
    """
    transform_list = []
    
    if target_size is not None:
        transform_list.append(T.Resize(target_size))
    
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    transform = T.Compose(transform_list)
    tensor = transform(img)
    
    # Add batch dimension
    return tensor.unsqueeze(0)

def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a normalized PyTorch tensor back to a PIL Image.

    Args:
        tensor (torch.Tensor): A 4D tensor of shape (1, 3, H, W).

    Returns:
        Image.Image: The denormalized PIL Image.
    """
    # Remove batch dimension
    tensor = tensor.squeeze(0).cpu()
    
    # Denormalize: x_orig = x_norm * std + mean
    # Reshape mean and std for broadcasting over (C, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    tensor = tensor * std + mean
    
    # Clamp to [0, 1] range to handle potential floating point errors
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    return T.ToPILImage()(tensor)

def concat_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Concatenates two PIL images side-by-side.
    Ensures both are converted to RGB mode first.
    """
    img1_rgb = img1.convert('RGB')
    img2_rgb = img2.convert('RGB')
    
    w1, h1 = img1_rgb.size
    w2, h2 = img2_rgb.size
    
    dst = Image.new('RGB', (w1 + w2, max(h1, h2)))
    dst.paste(img1_rgb, (0, 0))
    dst.paste(img2_rgb, (w1, 0))
    return dst
