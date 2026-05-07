import torch
import torch.optim as optim
import torch.nn.functional as F
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
from typing import Optional, List
import traceback

from .extractor import FeatureExtractor
from .transforms import preprocess_image, postprocess_tensor
from .utils import get_device

class OptimizationWorker(QThread):
    """
    Worker thread to perform Metamer optimization in the background.
    Communicates progress and results back to the GUI via signals.
    """
    
    # Signals
    progress = pyqtSignal(int, float)  # (iteration, loss)
    intermediate_image = pyqtSignal(object) # (current PIL Image)
    finished = pyqtSignal(object)      # (final PIL Image)
    error = pyqtSignal(str)            # (error message)

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str,
        target_image: Image.Image,
        starting_image: Optional[Image.Image] = None,
        mask_tensor: Optional[torch.Tensor] = None,
        lr: float = 0.1,
        iters: int = 300,
        loss_type: str = "mse",
        tv_weight: float = 0.0
    ):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.target_image = target_image
        self.starting_image = starting_image
        self.mask_tensor = mask_tensor
        self.lr = lr
        self.iters = iters
        self.loss_type = loss_type
        self.tv_weight = tv_weight
        self._stop_requested = False

    def request_stop(self):
        """Signals the optimization loop to stop gracefully."""
        self._stop_requested = True

    def run(self):
        extractor = None
        try:
            device = get_device()
            self.model.to(device)
            self.model.eval()

            # 1. Preprocess Target Image and Extract Reference Features
            target_tensor = preprocess_image(self.target_image, target_size=(224, 224)).to(device)
            
            extractor = FeatureExtractor(self.model, [self.target_layer])
            
            with torch.no_grad():
                target_features = extractor(target_tensor)
                target_act = target_features[self.target_layer].clone().detach()

            # 2. Prepare Starting Image and Apply Mask Initialization
            if self.starting_image:
                image_tensor = preprocess_image(self.starting_image, target_size=(224, 224)).to(device)
            else:
                image_tensor = torch.randn_like(target_tensor) * 0.01
            
            mask = None
            if self.mask_tensor is not None:
                mask = F.interpolate(self.mask_tensor.to(device), size=image_tensor.shape[2:], mode='nearest')
                image_tensor = image_tensor * mask + target_tensor * (1.0 - mask)

            image_tensor = image_tensor.detach().requires_grad_(True)

            # 3. Apply Mask Hook if provided
            if mask is not None:
                image_tensor.register_hook(lambda grad: grad * mask)

            # 4. Optimization Loop
            optimizer = optim.Adam([image_tensor], lr=self.lr)
            
            # We use MetamerOptimizer's static methods for logic consistency
            from .optimizer import MetamerOptimizer

            for i in range(self.iters):
                if self._stop_requested:
                    break

                optimizer.zero_grad()
                
                current_features = extractor(image_tensor)
                current_act = current_features[self.target_layer]
                
                # Calculate Core Loss
                if self.loss_type == "mse":
                    feature_loss = F.mse_loss(current_act, target_act)
                elif self.loss_type == "gram":
                    target_gram = MetamerOptimizer._calc_gram_matrix(target_act)
                    current_gram = MetamerOptimizer._calc_gram_matrix(current_act)
                    feature_loss = F.mse_loss(current_gram, target_gram)

                # Total Loss = Feature Loss + TV Regularization
                total_loss = feature_loss + self.tv_weight * MetamerOptimizer._calc_tv_loss(image_tensor)
                
                total_loss.backward()
                optimizer.step()
                
                # Emit progress to GUI
                self.progress.emit(i + 1, total_loss.item())
                
                # Live Update: Every 10 iterations
                if (i + 1) % 10 == 0:
                    current_img = postprocess_tensor(image_tensor.detach())
                    self.intermediate_image.emit(current_img)

            # 5. Postprocess and Finish
            final_img = postprocess_tensor(image_tensor.detach())
            self.finished.emit(final_img)

        except Exception as e:
            # Capture full traceback for debugging
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            
        finally:
            if extractor:
                extractor.remove_hooks()
