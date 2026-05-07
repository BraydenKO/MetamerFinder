import torch
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPoint, QSize, QRect
from PIL import Image
import numpy as np
from typing import Optional, List
from enum import Enum

class ToolMode(Enum):
    BRUSH = "brush"
    RECTANGLE = "rectangle"

class ImageMaskCanvas(QLabel):
    """
    A widget for displaying an image and drawing a transparency mask over it.
    The drawn areas represent 'frozen' pixels during optimization.
    Supports Freehand (Brush) and Rectangle tools with Undo functionality.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

        # Original image and pixmap
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        
        # Mask state
        self.mask_image: Optional[QImage] = None
        self.temp_rect: Optional[QRect] = None  # For visual feedback during rectangle drag
        self.start_point: Optional[QPoint] = None # For both brush and rectangle
        self.last_point: Optional[QPoint] = None
        
        # Tool Configuration
        self.tool_mode = ToolMode.BRUSH
        self.brush_size = 20
        self.brush_color = QColor(255, 0, 0, 100)  # Semi-transparent red
        
        # Undo Stack
        self._history: List[QImage] = []
        self.max_history = 20

    def set_tool_mode(self, mode: ToolMode):
        """Sets the current drawing tool."""
        self.tool_mode = mode

    def save_state(self):
        """Saves the current mask to the history stack for undo."""
        if self.mask_image:
            self._history.append(self.mask_image.copy())
            if len(self._history) > self.max_history:
                self._history.pop(0)

    def undo(self):
        """Restores the mask to the previous state."""
        if self._history:
            self.mask_image = self._history.pop()
            self._update_display()

    def load_image(self, pil_image: Image.Image):
        """
        Loads a PIL image, scales it to fit the widget, and initializes the mask.
        """
        # Convert PIL to QImage
        img_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qimage = QImage(img_data, pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)
        self.original_pixmap = QPixmap.fromImage(qimage)
        
        # Initialize the mask as a transparent image of the same size
        self.mask_image = QImage(qimage.size(), QImage.Format.Format_ARGB32)
        self.mask_image.fill(Qt.GlobalColor.transparent)
        
        self._history.clear()
        self._update_display()

    def _update_display(self, temp_draw: bool = False):
        """Scales the image and mask to fit the current widget size and displays them."""
        if self.original_pixmap is None:
            return

        # Scale original pixmap to fit widget
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Composite the mask over a copy of the scaled pixmap
        result_pixmap = QPixmap(scaled_pixmap.size())
        result_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(result_pixmap)
        painter.drawPixmap(0, 0, scaled_pixmap)
        
        # Draw the permanent mask
        scaled_mask = QPixmap.fromImage(self.mask_image).scaled(
            scaled_pixmap.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled_mask)
        
        # Draw temporary visual feedback (e.g., rectangle outline)
        if temp_draw and self.temp_rect:
            painter.setPen(QPen(self.brush_color, 2, Qt.PenStyle.DashLine))
            # Map temp_rect (in image coords) back to display coords
            ratio = scaled_pixmap.width() / self.original_pixmap.width()
            display_rect = QRect(
                int(self.temp_rect.x() * ratio),
                int(self.temp_rect.y() * ratio),
                int(self.temp_rect.width() * ratio),
                int(self.temp_rect.height() * ratio)
            )
            painter.drawRect(display_rect)
            
        painter.end()
        
        self.setPixmap(result_pixmap)
        self.display_pixmap = scaled_pixmap

    def clear_mask(self):
        """Eraser all drawings on the mask."""
        if self.mask_image:
            self.save_state()
            self.mask_image.fill(Qt.GlobalColor.transparent)
            self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.save_state()
            self.start_point = self._map_to_image_coords(event.position().toPoint())
            self.last_point = self.start_point
            
            if self.tool_mode == ToolMode.BRUSH:
                self._draw_on_mask(self.start_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.start_point:
            current_point = self._map_to_image_coords(event.position().toPoint())
            
            if self.tool_mode == ToolMode.BRUSH:
                self._draw_on_mask(current_point, self.last_point)
                self.last_point = current_point
            elif self.tool_mode == ToolMode.RECTANGLE:
                self.temp_rect = QRect(self.start_point, current_point).normalized()
                self._update_display(temp_draw=True)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool_mode == ToolMode.RECTANGLE and self.start_point:
                current_point = self._map_to_image_coords(event.position().toPoint())
                final_rect = QRect(self.start_point, current_point).normalized()
                self._commit_rect_to_mask(final_rect)
                self.temp_rect = None
            
            self.start_point = None
            self.last_point = None
            self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _map_to_image_coords(self, pos: QPoint) -> QPoint:
        """Maps widget coordinates to original image coordinates."""
        if not self.display_pixmap:
            return pos
        offset_x = (self.width() - self.display_pixmap.width()) // 2
        offset_y = (self.height() - self.display_pixmap.height()) // 2
        local_x = pos.x() - offset_x
        local_y = pos.y() - offset_y
        ratio = self.original_pixmap.width() / self.display_pixmap.width()
        return QPoint(int(local_x * ratio), int(local_y * ratio))

    def _draw_on_mask(self, end_point: QPoint, start_point: Optional[QPoint] = None):
        """Draws brush strokes on the internal mask image."""
        if not self.mask_image:
            return
        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.brush_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if start_point:
            painter.drawLine(start_point, end_point)
        else:
            painter.drawPoint(end_point)
        painter.end()
        self._update_display()

    def _commit_rect_to_mask(self, rect: QRect):
        """Draws a filled rectangle on the internal mask image."""
        if not self.mask_image:
            return
        painter = QPainter(self.mask_image)
        painter.setBrush(QBrush(self.brush_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)
        painter.end()

    def get_mask_tensor(self) -> torch.Tensor:
        """
        Converts the mask to a PyTorch tensor of shape (1, 1, H, W).
        1.0 = Free to change (transparent areas).
        0.0 = Frozen (drawn areas).
        """
        if self.mask_image is None:
            return torch.ones((1, 1, 224, 224))
        ptr = self.mask_image.bits()
        ptr.setsize(self.mask_image.height() * self.mask_image.width() * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((self.mask_image.height(), self.mask_image.width(), 4))
        alpha = arr[:, :, 3]
        mask_data = np.where(alpha > 50, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0)
