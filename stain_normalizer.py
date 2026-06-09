from torchstain.normalizers import MacenkoNormalizer
from PIL import Image
from skimage.color import rgb2hsv
import torch
import numpy as np


class WSI_normalizer:

    def __init__(self, reference_tile):
        self.normalizer = MacenkoNormalizer()
    

    def has_pen_marks(self, tile: Image.Image, threshold=0.02) -> bool:
        """Detect annotation pen marks by looking for non-H&E colors in HSV."""
        arr = np.array(tile) / 255.0
        hsv = rgb2hsv(arr)
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        
        # H&E hues are purple/pink (0.7-1.0 and 0.0-0.1) and pink (0.85-1.0)
        # Pen marks are typically green (0.25-0.45) or red (0.95-1.0) or blue (0.55-0.7)
        is_green = ((hue > 0.25) & (hue < 0.45) & (sat > 0.3))
        is_blue  = ((hue > 0.55) & (hue < 0.70) & (sat > 0.3))
        is_red   = ((hue > 0.95) | (hue < 0.05)) & (sat > 0.5)
        
        pen_ratio = (is_green | is_blue | is_red).mean()
        return pen_ratio > threshold
    
    
