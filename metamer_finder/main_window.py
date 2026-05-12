import sys
import os
import torch
import torchvision.models as models
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
    QLabel, QFrame, QProgressBar, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSlot
from PIL import Image

# Import internal components
from .canvas import ImageMaskCanvas, ToolMode
from .worker import OptimizationWorker
from .transforms import concat_images
from .utils import get_device, load_model
from .inspector import analyze_skip_connections

class MainWindow(QMainWindow):
    """
    Main Window for the Metamer Finder Application.
    Provides a GUI for loading models, selecting layers, and managing optimization.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metamer Finder")
        self.resize(1200, 800)
        
        # State variables
        self.current_model = None
        self.skip_analysis = {}
        self.target_image = None
        self.target_data_path = None
        self.custom_model_path = None
        self.custom_loss_path = None
        self.worker = None
        
        self._setup_ui()

    def _setup_ui(self):
        """Initializes the layout and UI components."""
        # Central widget and main horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Sidebar (Controls) ---
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 1. Data Management
        sidebar_layout.addWidget(QLabel("<b>1. Data Management</b>"))
        self.load_image_btn = QPushButton("Load Target (Image/Tensor)")
        self.load_image_btn.clicked.connect(self._open_data)
        sidebar_layout.addWidget(self.load_image_btn)
        
        sidebar_layout.addSpacing(10)
        
        # 2. Model Management
        sidebar_layout.addWidget(QLabel("<b>2. Model Architecture</b>"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Select a model...", "VGG16", "ResNet18", "ResNet50", "ViT-B/16"])
        self.model_dropdown.currentTextChanged.connect(self._on_model_changed)
        sidebar_layout.addWidget(self.model_dropdown)
        
        sidebar_layout.addSpacing(10)

        # 2b. Custom Assets
        sidebar_layout.addWidget(QLabel("<b>2b. Custom Assets</b>"))
        self.load_custom_model_btn = QPushButton("Load Custom Model (.py / .pth)")
        self.load_custom_model_btn.clicked.connect(self._load_custom_model_script)
        sidebar_layout.addWidget(self.load_custom_model_btn)

        self.load_custom_loss_btn = QPushButton("Load Custom Loss Script")
        self.load_custom_loss_btn.clicked.connect(self._load_custom_loss_script)
        sidebar_layout.addWidget(self.load_custom_loss_btn)

        sidebar_layout.addSpacing(10)
        
        # 3. Layer Selection
        sidebar_layout.addWidget(QLabel("<b>3. Target Layer</b>"))
        self.layer_dropdown = QComboBox()
        sidebar_layout.addWidget(self.layer_dropdown)
        
        sidebar_layout.addSpacing(10)
        
        # 4. Starting State
        sidebar_layout.addWidget(QLabel("<b>4. Starting State</b>"))
        self.start_image_dropdown = QComboBox()
        self.start_image_dropdown.addItems(["Random Noise", "Base Image"])
        sidebar_layout.addWidget(self.start_image_dropdown)
        
        sidebar_layout.addSpacing(20)
        
        # 5. Hyperparameters
        sidebar_layout.addWidget(QLabel("<b>5. Hyperparameters</b>"))
        
        # Matching Strategy
        sidebar_layout.addWidget(QLabel("Matching Strategy:"))
        self.loss_type_dropdown = QComboBox()
        self.loss_type_dropdown.addItem("Exact Spatial (MSE)", "mse")
        self.loss_type_dropdown.addItem("Texture/Style (Gram Matrix)", "gram")
        sidebar_layout.addWidget(self.loss_type_dropdown)

        # TV Smoothing
        sidebar_layout.addWidget(QLabel("TV Smoothing Weight:"))
        self.tv_spinbox = QDoubleSpinBox()
        self.tv_spinbox.setRange(0.0, 1.0)
        self.tv_spinbox.setDecimals(6)
        self.tv_spinbox.setSingleStep(0.001)
        self.tv_spinbox.setValue(0.0)
        sidebar_layout.addWidget(self.tv_spinbox)

        # Learning Rate
        sidebar_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.000001, 10.0)
        self.lr_spinbox.setDecimals(6)
        self.lr_spinbox.setSingleStep(0.001)
        self.lr_spinbox.setValue(0.1)
        sidebar_layout.addWidget(self.lr_spinbox)
        
        # Iterations
        sidebar_layout.addWidget(QLabel("Max Iterations:"))
        self.iters_spinbox = QSpinBox()
        self.iters_spinbox.setRange(10, 10000)
        self.iters_spinbox.setValue(300)
        self.iters_spinbox.setSingleStep(50)
        sidebar_layout.addWidget(self.iters_spinbox)
        
        sidebar_layout.addSpacing(30)
        
        # 6. Progress and Actions
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        sidebar_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        sidebar_layout.addWidget(self.status_label)
        
        self.optimize_btn = QPushButton("Start Optimization")
        self.optimize_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        self.optimize_btn.clicked.connect(self._start_optimization)
        sidebar_layout.addWidget(self.optimize_btn)

        self.stop_btn = QPushButton("Stop Optimization")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_optimization)
        sidebar_layout.addWidget(self.stop_btn)
        
        # --- Drawing Canvas Area ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        
        canvas_layout.addWidget(QLabel("<b>Input & Masking Canvas</b>"))
        
        self.canvas = ImageMaskCanvas()
        
        # Drawing Toolbar
        drawing_toolbar = QHBoxLayout()
        self.brush_btn = QPushButton("Brush Tool")
        self.brush_btn.clicked.connect(lambda: self.canvas.set_tool_mode(ToolMode.BRUSH))
        
        self.rect_btn = QPushButton("Rectangle Tool")
        self.rect_btn.clicked.connect(lambda: self.canvas.set_tool_mode(ToolMode.RECTANGLE))
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.canvas.undo)
        
        self.clear_mask_btn_toolbar = QPushButton("Clear Mask")
        self.clear_mask_btn_toolbar.clicked.connect(self.canvas.clear_mask)
        
        drawing_toolbar.addWidget(self.brush_btn)
        drawing_toolbar.addWidget(self.rect_btn)
        drawing_toolbar.addWidget(self.undo_btn)
        drawing_toolbar.addWidget(self.clear_mask_btn_toolbar)
        drawing_toolbar.addStretch()
        
        canvas_layout.addLayout(drawing_toolbar)
        canvas_layout.addWidget(self.canvas, stretch=1)
        
        # --- Live Result Area ---
        self.live_result_label = QLabel("Optimization result will appear here...")
        self.live_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_result_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.live_result_label.setMinimumSize(224, 224)
        
        canvas_layout.addWidget(QLabel("<b>Live Optimization Result</b>"))
        canvas_layout.addWidget(self.live_result_label, stretch=1)
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(canvas_container, stretch=1)

    def _open_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data (Image or Tensor)", "inputs", 
            "All Supported (*.png *.jpg *.jpeg *.bmp *.npy *.pt *.pth);;Image Files (*.png *.jpg *.jpeg *.bmp);;Tensor Files (*.npy *.pt *.pth)"
        )
        if file_path:
            self.target_data_path = file_path
            from .utils import is_image_file
            if is_image_file(file_path):
                self.target_image = Image.open(file_path).convert("RGB")
                self.canvas.load_image(self.target_image)
            else:
                self.target_image = None
                self.canvas.clear_image()
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")

    def _load_custom_model_script(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model (.py or .pth)", "", 
            "Model Files (*.py *.pth *.pt)"
        )
        if file_path:
            self.custom_model_path = file_path
            self.model_dropdown.setCurrentIndex(0)
            
            if file_path.lower().endswith(('.pth', '.pt')):
                print(f"Loading custom model weights from {file_path}")
                self.status_label.setText(f"Loading {os.path.basename(file_path)}...")
                try:
                    self.current_model = load_model(file_path, get_device())
                    print("Analyzing model architecture...")
                    self.skip_analysis = analyze_skip_connections(self.current_model)
                    self._populate_layer_dropdown()
                    self.status_label.setText(f"Custom Model Loaded: {os.path.basename(file_path)}")
                    # For .pth files, we've loaded the model, so we can clear the path
                    # and let the worker use self.current_model
                    self.custom_model_path = None
                    self.model_dropdown.setEnabled(True)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load custom model: {str(e)}")
                    self.status_label.setText("Custom Model load failed.")
                    self.model_dropdown.setEnabled(True)
            else:
                self.current_model = None
                self.status_label.setText(f"Custom Model Script: {os.path.basename(file_path)}")
                self.model_dropdown.setEnabled(False)
                # For script files, we can't analyze until run time
                self.layer_dropdown.clear()
                self.layer_dropdown.addItem("Dynamic (enter layer name in code/terminal)", "custom")

    def _load_custom_loss_script(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Loss Script", "", "Python Files (*.py)"
        )
        if file_path:
            self.custom_loss_path = file_path
            self.status_label.setText(f"Custom Loss Script: {os.path.basename(file_path)}")
        else:
            # Clear if canceled? Or maybe add a dedicated clear button.
            # For now, let's allow clearing by canceling if it was already set.
            if self.custom_loss_path:
                if QMessageBox.question(self, "Clear Loss?", "Do you want to clear the custom loss script?") == QMessageBox.StandardButton.Yes:
                    self.custom_loss_path = None
                    self.status_label.setText("Custom Loss Script Cleared.")

    def _on_model_changed(self, model_name: str):
        """Loads the selected model and updates the layer dropdown."""
        if model_name == "Select a model...":
            self.layer_dropdown.clear()
            self.current_model = None
            return

        # Clear custom model path if we are switching to a standard model
        self.custom_model_path = None
        
        print(f"Loading {model_name}...")
        self.status_label.setText(f"Loading {model_name}...")
        self.model_dropdown.setEnabled(False)
        
        try:
            # Instantiate model
            model_id_map = {
                "VGG16": "vgg16",
                "ResNet18": "resnet18",
                "ResNet50": "resnet50",
                "ViT-B/16": "vit_b_16"
            }
            
            if model_name in model_id_map:
                self.current_model = load_model(model_id_map[model_name], get_device())
            
            # Analyze skip connections
            print("Analyzing model architecture...")
            self.skip_analysis = analyze_skip_connections(self.current_model)
            
            # Populate dropdown
            self._populate_layer_dropdown()
            
            self.status_label.setText(f"{model_name} Loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_label.setText("Model load failed.")
        finally:
            self.model_dropdown.setEnabled(True)

    def _populate_layer_dropdown(self):
        """Fills the combo box with model layers and safety indicators."""
        self.layer_dropdown.clear()
        if not self.current_model:
            return
            
        for name, _ in self.current_model.named_modules():
            if name == "": continue
            
            is_inside_residual = self.skip_analysis.get(name, False)
            if is_inside_residual:
                display_text = f"{name} ⚠️ (Inside Residual)"
            else:
                display_text = f"{name} ✅ (Safe Choke Point)"
                
            self.layer_dropdown.addItem(display_text, name)

    def _start_optimization(self):
        """Prepares and starts the background optimization worker."""
        if not self.target_image and not self.target_data_path:
            QMessageBox.warning(self, "Warning", "Please load target data first.")
            return
        if not self.current_model and not self.custom_model_path:
            QMessageBox.warning(self, "Warning", "Please select a model or load a custom model script.")
            return
        if self.layer_dropdown.currentIndex() == -1:
            QMessageBox.warning(self, "Warning", "Please select a target layer.")
            return

        # Prepare parameters
        target_layer = self.layer_dropdown.currentData()
        lr = self.lr_spinbox.value()
        iters = self.iters_spinbox.value()
        loss_type = self.loss_type_dropdown.currentData()
        tv_weight = self.tv_spinbox.value()
        
        # Starting image logic
        start_choice = self.start_image_dropdown.currentText()
        starting_image = self.target_image if start_choice == "Base Image" else None
        
        # Get mask from canvas
        mask_tensor = self.canvas.get_mask_tensor() if self.target_image else None
        
        # Setup UI for optimization
        self.progress_bar.setMaximum(iters)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.optimize_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_image_btn.setEnabled(False)
        self.model_dropdown.setEnabled(False)
        self.load_custom_model_btn.setEnabled(False)
        self.load_custom_loss_btn.setEnabled(False)
        self.status_label.setText("Optimizing...")

        # Initialize Worker
        self.worker = OptimizationWorker(
            model=self.current_model,
            target_layer=target_layer,
            target_image=self.target_image,
            target_data_path=self.target_data_path,
            custom_model_path=self.custom_model_path,
            custom_loss_path=self.custom_loss_path,
            starting_image=starting_image,
            mask_tensor=mask_tensor,
            lr=lr,
            iters=iters,
            loss_type=loss_type,
            tv_weight=tv_weight
        )
        
        # Connect signals
        self.worker.progress.connect(self._on_optimization_progress)
        self.worker.intermediate_image.connect(self._on_intermediate_image)
        self.worker.finished.connect(self._on_optimization_finished)
        self.worker.error.connect(self._on_optimization_error)
        
        # Start
        self.worker.start()

    def _stop_optimization(self):
        """Terminates the background worker early and gracefully."""
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            # We wait a bit for it to finish the current iteration and exit
            self.worker.wait()
            self._cleanup_optimization_ui()
            self.status_label.setText("Optimization Stopped.")

    def _cleanup_optimization_ui(self):
        """Resets the UI elements after optimization ends or is stopped."""
        self.progress_bar.setVisible(False)
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_image_btn.setEnabled(True)
        self.model_dropdown.setEnabled(True)
        self.load_custom_model_btn.setEnabled(True)
        self.load_custom_loss_btn.setEnabled(True)

    @pyqtSlot(int, float)
    def _on_optimization_progress(self, iteration, loss):
        self.progress_bar.setValue(iteration)
        self.status_label.setText(f"Iteration {iteration}/{self.iters_spinbox.value()} | Loss: {loss:.6f}")

    @pyqtSlot(object)
    def _on_intermediate_image(self, pil_image: Image.Image):
        self._update_live_result(pil_image)

    @pyqtSlot(object)
    def _on_optimization_finished(self, result):
        self._cleanup_optimization_ui()
        self.status_label.setText("Optimization Complete.")
        
        if isinstance(result, Image.Image):
            # Update final live result
            self._update_live_result(result)
            
            # Quick preview
            result.show() 
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metamer", "results", "JPEG (*.jpg);;PNG (*.png)"
            )
            if save_path:
                # 1. Save the main metamer
                result.save(save_path)
                
                # 2. Generate and save the comparison image
                try:
                    base, ext = os.path.splitext(save_path)
                    compare_path = f"{base}_compare{ext}"
                    
                    if self.target_image:
                        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
                        target_resized = self.target_image.resize(result.size, resample_filter)
                        
                        comparison_img = concat_images(target_resized, result)
                        comparison_img.save(compare_path)
                        
                        QMessageBox.information(
                            self, "Saved", 
                            f"Metamer saved to:\n{save_path}\n\nComparison saved to:\n{compare_path}"
                        )
                    else:
                        QMessageBox.information(self, "Saved", f"Metamer saved to {save_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Comparison Error", f"Metamer saved, but comparison failed: {str(e)}")
        else:
            # Result is a Tensor
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metamer Tensor", "results", "Torch Tensor (*.pt);;Numpy Array (*.npy)"
            )
            if save_path:
                from .utils import save_data
                if save_path.endswith('.npy'):
                    import numpy as np
                    np.save(save_path, result.numpy())
                else:
                    save_data(result, save_path)
                QMessageBox.information(self, "Saved", f"Metamer tensor saved to {save_path}")

    @pyqtSlot(str)
    def _on_optimization_error(self, error_msg):
        self._cleanup_optimization_ui()
        self.status_label.setText("Optimization Error.")
        QMessageBox.critical(self, "Optimization Error", error_msg)

    def _update_live_result(self, pil_image: Image.Image):
        """Converts PIL image to QPixmap and displays it in live_result_label."""
        # Convert PIL to QImage
        pil_image = pil_image.convert("RGBA")
        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)
        
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self.live_result_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.live_result_label.setPixmap(scaled_pixmap)
