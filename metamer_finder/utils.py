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
    # Case 1: Load from local file
    if os.path.exists(model_id):
        print(f"Loading custom model from: {model_id}")
        try:
            # We use weights_only=False to support loading full nn.Module objects.
            # Users should only load models from trusted sources.
            model = torch.load(model_id, map_location=device, weights_only=False)
            if not isinstance(model, torch.nn.Module):
                raise ValueError("The provided file does not contain a full nn.Module.")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            sys.exit(1)
    
    # Case 2: Load from torchvision
    else:
        print(f"Loading torchvision model: {model_id}")
        try:
            model_builder = getattr(models, model_id)
            weights_enum = getattr(models, f"{model_id.upper()}_Weights", None)
            if weights_enum:
                model = model_builder(weights=weights_enum.DEFAULT)
            else:
                model = model_builder(pretrained=True)
        except AttributeError:
            print(f"Error: '{model_id}' is not a valid torchvision model name or file path.")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing torchvision model: {e}")
            sys.exit(1)

    return model.to(device).eval()

def is_image_file(filename: str) -> bool:
    """Checks if a filename has an image extension."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    return filename.lower().endswith(extensions)

def load_data(path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads data from a file path. If it's a .pt or .pth file, loads as tensor.
    Otherwise, assumes it will be handled by the vision pipeline.
    """
    if path.lower().endswith(('.pt', '.pth')):
        data = torch.load(path, map_location=device, weights_only=False)
        if isinstance(data, torch.Tensor):
            # Ensure it has a batch dimension if it's a vector
            if data.dim() == 1:
                data = data.unsqueeze(0)
            return data
    return None

def save_data(data: torch.Tensor, path: str):
    """Saves a tensor to a .pt file."""
    torch.save(data.cpu(), path)
