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
    finished = pyqtSignal(object)      # (final PIL Image or Tensor)
    error = pyqtSignal(str)            # (error message)

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str,
        target_image: Optional[Image.Image] = None,
        target_data_path: Optional[str] = None,
        custom_model_path: Optional[str] = None,
        custom_loss_path: Optional[str] = None,
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
        self.target_data_path = target_data_path
        self.custom_model_path = custom_model_path
        self.custom_loss_path = custom_loss_path
        self.starting_image = starting_image
        self.mask_tensor = mask_tensor
        self.lr = lr
        self.iters = iters
        self.loss_type = loss_type
        self.tv_weight = tv_weight
        self._stop_requested = False
        self.is_vision = True

    def _load_custom_script(self, path: str, func_name: str):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, func_name):
            return getattr(module, func_name)
        raise AttributeError(f"Module at {path} does not have a '{func_name}' function.")

    def request_stop(self):
        """Sets the stop flag to break the optimization loop."""
        self._stop_requested = True

    def run(self):
        extractor = None
        try:
            device = get_device()
            
            # 1. Resolve Model
            if self.custom_model_path:
                if self.custom_model_path.lower().endswith(('.pth', '.pt')):
                    print(f"Loading custom model weights from {self.custom_model_path}")
                    from .utils import load_model
                    self.model = load_model(self.custom_model_path, device)
                else:
                    print(f"Loading custom model script from {self.custom_model_path}")
                    load_func = self._load_custom_script(self.custom_model_path, "get_model")
                    self.model = load_func().to(device)
            else:
                self.model.to(device)
            
            self.model.eval()

            # 2. Resolve Custom Loss
            custom_loss_hook = None
            if self.custom_loss_path:
                print(f"Loading custom loss script from {self.custom_loss_path}")
                custom_loss_hook = self._load_custom_script(self.custom_loss_path, "custom_penalty")

            # 3. Load and Preprocess Target Data
            from .utils import load_data, is_image_file
            if self.target_data_path:
                self.is_vision = is_image_file(self.target_data_path)
            
            if self.is_vision:
                target_tensor = preprocess_image(self.target_image, target_size=(224, 224)).to(device)
            else:
                target_tensor = load_data(self.target_data_path, device)
            
            extractor = FeatureExtractor(self.model, [self.target_layer])
            
            with torch.no_grad():
                target_features = extractor(target_tensor)
                target_act = target_features[self.target_layer].clone().detach()

            # 4. Prepare Starting Data
            if self.starting_image and self.is_vision:
                image_tensor = preprocess_image(self.starting_image, target_size=(224, 224)).to(device)
            else:
                image_tensor = torch.randn_like(target_tensor) * 0.01
            
            mask = None
            if self.mask_tensor is not None and self.is_vision:
                mask = F.interpolate(self.mask_tensor.to(device), size=image_tensor.shape[2:], mode='nearest')
                image_tensor = image_tensor * mask + target_tensor * (1.0 - mask)

            image_tensor = image_tensor.detach().requires_grad_(True)

            # 5. Apply Mask Hook if provided
            if mask is not None:
                image_tensor.register_hook(lambda grad: grad * mask)

            # 6. Optimization Loop
            optimizer = optim.Adam([image_tensor], lr=self.lr)
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
                
                # Apply custom loss hook
                if custom_loss_hook:
                    total_loss += custom_loss_hook(image_tensor, current_features, target_features)
                
                total_loss.backward()
                optimizer.step()
                
                # Emit progress to GUI
                self.progress.emit(i + 1, total_loss.item())
                
                # Live Update: Every 10 iterations (Image only)
                if (i + 1) % 10 == 0 and self.is_vision:
                    current_img = postprocess_tensor(image_tensor.detach())
                    self.intermediate_image.emit(current_img)

            # 7. Postprocess and Finish
            if self.is_vision:
                final_res = postprocess_tensor(image_tensor.detach())
            else:
                final_res = image_tensor.detach().cpu()
                
            self.finished.emit(final_res)

        except Exception as e:
            # Capture full traceback for debugging
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            
        finally:
            if extractor:
                extractor.remove_hooks()
