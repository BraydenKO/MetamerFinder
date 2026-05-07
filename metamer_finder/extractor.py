import torch
import torch.nn as nn
from typing import List, Dict, Any

class FeatureExtractor:
    """
    A class to extract intermediate activations from a PyTorch model using forward hooks.

    Attributes:
        model (nn.Module): The PyTorch model to extract features from.
        target_layers (List[str]): Names of the layers to hook into.
        features (Dict[str, torch.Tensor]): Dictionary storing captured activations.
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initializes the FeatureExtractor and registers hooks.

        Args:
            model (nn.Module): The PyTorch model.
            target_layers (List[str]): List of strings representing the layer names.
                                       These must match names in model.named_modules().
        
        Raises:
            ValueError: If a target layer name is not found in the model.
        """
        self.model = model
        self.target_layers = target_layers
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _get_hook(self, layer_name: str):
        """Creates a forward hook function for a specific layer."""
        def hook(module: nn.Module, input: Any, output: torch.Tensor):
            # Store the output tensor. We don't clone here to save memory, 
            # but users should be aware that these tensors are part of the graph.
            self.features[layer_name] = output
        return hook

    def _register_hooks(self):
        """Iterates through target layers and registers forward hooks."""
        module_dict = dict(self.model.named_modules())
        
        for name in self.target_layers:
            if name not in module_dict:
                self.remove_hooks()  # Clean up any already registered hooks
                raise ValueError(f"Layer '{name}' not found in the model. "
                                 f"Available layers: {list(module_dict.keys())}")
            
            layer = module_dict[name]
            hook = layer.register_forward_hook(self._get_hook(name))
            self._hooks.append(hook)

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass on the model and captures target activations.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping layer names to their activation tensors.
        """
        self.features = {}  # Clear previous activations
        self.model(x)
        return self.features

    def remove_hooks(self):
        """
        Safely removes all registered hooks to prevent memory leaks and unexpected behavior.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        """Ensures hooks are removed when the object is garbage collected."""
        self.remove_hooks()
