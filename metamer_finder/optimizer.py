import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Any
from tqdm import tqdm
from .extractor import FeatureExtractor

class MetamerOptimizer:
    """
    Core optimization loop for generating metamers by matching intermediate activations.
    Supports Exact Spatial matching (MSE) and Texture matching (Gram Matrix),
    with optional Total Variation (TV) regularization.

    Attributes:
        extractor (FeatureExtractor): The feature extractor wrapping the target model.
        target_features (Dict[str, torch.Tensor]): The goal activations to match.
        lr (float): Learning rate for the Adam optimizer.
        max_iterations (int): Maximum number of optimization steps.
        tv_weight (float): Weight for Total Variation smoothing.
        loss_type (str): Optimization strategy ('mse' or 'gram').
    """

    def __init__(
        self, 
        extractor: FeatureExtractor, 
        target_features: Dict[str, torch.Tensor], 
        lr: float = 0.01, 
        max_iterations: int = 500,
        tv_weight: float = 0.0,
        loss_type: str = "mse",
        custom_loss_hook: Optional[Any] = None
    ):
        """
        Initializes the MetamerOptimizer.

        Args:
            extractor (FeatureExtractor): Pre-initialized FeatureExtractor.
            target_features (Dict[str, torch.Tensor]): Target activations.
            lr (float): Adam learning rate. Defaults to 0.01.
            max_iterations (int): Optimization steps. Defaults to 500.
            tv_weight (float): Weight for Total Variation regularization. Defaults to 0.0.
            loss_type (str): 'mse' for exact spatial matching, 'gram' for texture matching.
            custom_loss_hook (Optional[Callable]): A function that takes the optimized tensor and returns a scalar loss.
        """
        self.extractor = extractor
        self.target_features = target_features
        self.lr = lr
        self.max_iterations = max_iterations
        self.tv_weight = tv_weight
        self.loss_type = loss_type
        self.custom_loss_hook = custom_loss_hook

    @staticmethod
    def _calc_tv_loss(tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates Total Variation loss dynamically based on tensor dimensions.
        """
        
        if tensor.dim() == 4:  # Image: (B, C, H, W)
            diff_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
            diff_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
            return diff_h + diff_w
        elif tensor.dim() == 3:  # 1D Sequence: (B, C, T)
            diff_t = torch.abs(tensor[:, :, 1:] - tensor[:, :, :-1]).mean()
            return diff_t
        
        return torch.tensor(0.0, device=tensor.device)

    @staticmethod
    def _calc_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix in a data-agnostic way.
        """
        b, c, *dims = tensor.size()
        # Flatten remaining dimensions (e.g., H*W for images, T for 1D)
        features = tensor.view(b, c, -1)  # (B, C, N)
        n = features.size(2)
        
        # Batch matrix multiplication: (B, C, N) @ (B, N, C) -> (B, C, C)
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by (C * N) to keep values stable across architectures
        return gram / (c * n)

    def generate(
        self, 
        starting_image: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates a metamer from the starting image.

        Args:
            starting_image (torch.Tensor): Initial input tensor of shape (B, C, ...).
            mask (Optional[torch.Tensor]): Mask of shape (B, 1, ...).
                                          1/True: pixel can change. 0/False: pixel is frozen.

        Returns:
            torch.Tensor: The optimized metamer tensor, detached from the graph.
        """
        # Prepare the input tensor for optimization
        image_tensor = starting_image.clone().detach().requires_grad_(True)

        # Apply gradient masking if provided
        if mask is not None:
            image_tensor.register_hook(lambda grad: grad * mask.to(grad.dtype))

        optimizer = optim.Adam([image_tensor], lr=self.lr)
        
        pbar = tqdm(range(self.max_iterations), desc=f"Optimizing ({self.loss_type})")
        
        for _ in pbar:
            optimizer.zero_grad()

            # Extract current features
            current_features = self.extractor(image_tensor)

            # Calculate total loss across all target layers
            feature_loss = 0.0
            for layer_name, target_act in self.target_features.items():
                current_act = current_features[layer_name]
                
                if self.loss_type == "mse":
                    feature_loss += F.mse_loss(current_act, target_act)
                elif self.loss_type == "gram":
                    target_gram = self._calc_gram_matrix(target_act)
                    current_gram = self._calc_gram_matrix(current_act)
                    feature_loss += F.mse_loss(current_gram, target_gram)

            # Total Loss = Feature Loss + TV Regularization
            total_loss = feature_loss + self.tv_weight * self._calc_tv_loss(image_tensor)

            # Apply custom loss hook if provided
            if self.custom_loss_hook is not None:
                total_loss += self.custom_loss_hook(image_tensor)

            total_loss.backward()
            optimizer.step()

            # Update progress bar with loss information
            pbar.set_postfix({"loss": f"{total_loss.item():.6f}"})

        return image_tensor.detach()
