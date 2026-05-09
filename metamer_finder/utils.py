import torch
import torchvision.models as models
import os
import sys

def get_device() -> torch.device:
    """Detects the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(model_id: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a model from torchvision or a local file path.
    
    Args:
        model_id: Name of a torchvision model (e.g., 'vgg16') or path to a .pth file.
        device: The device to load the model onto.
        
    Returns:
        The loaded and evaluation-ready nn.Module.
    """
    # Heuristic to check if model_id is likely a file path
    is_path = model_id.endswith(('.pt', '.pth')) or os.path.sep in model_id or (os.path.altsep and os.path.altsep in model_id)
    
    if is_path or os.path.exists(model_id):
        if not os.path.exists(model_id):
            raise FileNotFoundError(f"Model file not found at '{model_id}'")
            
        print(f"Loading custom model from: {model_id}")
        # We use weights_only=False to support loading full nn.Module objects.
        # Users should only load models from trusted sources.
        checkpoint = torch.load(model_id, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            # If it's a dictionary, it might be a state dict. 
            # We return it as is and let the caller decide what to do.
            # But the return type hint says nn.Module, so this is a bit tricky.
            # For now, let's keep the error but make it a specific exception.
            return checkpoint # Caller should check type
        else:
            raise ValueError(f"The provided file contains a {type(checkpoint)}, not an nn.Module or dict.")
    
    # Case 2: Load from torchvision
    else:
        print(f"Loading torchvision model: {model_id}")
        if hasattr(models, model_id):
            model_builder = getattr(models, model_id)
            # Try new weights API first
            weights_attr = f"{model_id.upper()}_Weights"
            if hasattr(models, weights_attr):
                weights_enum = getattr(models, weights_attr)
                model = model_builder(weights=weights_enum.DEFAULT)
            else:
                # Fallback for older torchvision or models without weights enum
                model = model_builder(pretrained=True)
        else:
            raise AttributeError(f"'{model_id}' is not a valid torchvision model name.")

    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    return model

def is_image_file(filename: str) -> bool:
    """Checks if a filename has an image extension."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    return filename.lower().endswith(extensions)

def load_data(path: str, device: torch.device) -> torch.Tensor:
    """
    Loads data from a file path. Supports .npy (numpy) and .pt/.pth (torch) files.
    Returns a torch.Tensor on the specified device.
    """
    import numpy as np
    
    path_lower = path.lower()
    
    if path_lower.endswith(('.pt', '.pth')):
        data = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(data, torch.Tensor):
            return None
    elif path_lower.endswith('.npy'):
        data = np.load(path)
        data = torch.from_numpy(data).to(device)
    else:
        return None

    # Ensure it has a batch dimension if it's a vector
    if data.dim() == 1:
        data = data.unsqueeze(0)
    
    return data.float()

def save_data(data: torch.Tensor, path: str):
    """Saves a tensor to a .pt file."""
    torch.save(data.cpu(), path)
