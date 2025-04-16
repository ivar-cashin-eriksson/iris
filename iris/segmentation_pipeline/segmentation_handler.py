"""
Handler for image segmentation using SAM2 model.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import hashlib

from iris.config.segmentation_pipeline_config_manager import SAM2Config
from iris.utils.machine_utils import get_device
from iris.utils.image_utils import convert_mask_format, convert_image_format
from iris.data_pipeline.mongodb_manager import MongoDBManager

class SegmentationHandler:
    """Handles image segmentation using SAM2 model."""
    
    def __init__(self, sam2_config: SAM2Config):
        """
        Initialize the SegmentationHandler.
        
        Args:
            sam2_config: SegmentationConfig object containing all 
                         configuration parameters
        """
        self.sam2_config = sam2_config
        self.device = get_device(self.sam2_config.device)
        self.target_size = 256  # Maximum size for the longest dimension
        
        # Configure CUDA settings if available
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Initialize SAM2 model
        self.sam2 = sam_model_registry[
            self.sam2_config.model_type
        ](checkpoint=self.sam2_config.get_checkpoint_path())
        self.sam2.to(self.device)
        self.sam2.eval()
        
        # Initialize mask generator with configuration parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam2,
            **self.sam2_config.mask_generator_params
        )
    
    def _downsample_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Downsample image while preserving aspect ratio.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            tuple: (downsampled image, original size)
        """
        height, width = image.shape[:2]
        original_size = (height, width)
        
        # Calculate scaling factor
        scale = self.target_size / max(height, width)
        if scale >= 1:  # Don't upscale
            return image, original_size
            
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Use PIL for high-quality downsampling
        pil_img = Image.fromarray(image)
        resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(resized_img), original_size

    def _upscale_masks(self, masks: List[Dict[str, Any]], original_size: tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Scale masks back to original image size.
        
        Args:
            masks: List of mask dictionaries
            original_size: Original image size (height, width)
            
        Returns:
            List of mask dictionaries with scaled masks and coordinates
        """
        height, width = original_size
        
        for mask in masks:
            # Scale segmentation mask
            mask_array = mask['segmentation']
            scaled_mask = Image.fromarray(mask_array).resize(
                (width, height),
                Image.Resampling.NEAREST
            )
            mask['segmentation'] = np.array(scaled_mask)
            
            # Scale bounding box
            x, y, w, h = mask['bbox']
            scale_h = height / mask_array.shape[0]
            scale_w = width / mask_array.shape[1]
            mask['bbox'] = [
                int(x * scale_w),
                int(y * scale_h),
                int(w * scale_w),
                int(h * scale_h)
            ]
            
            # Scale point coordinates if they exist
            if 'point_coords' in mask and mask['point_coords'] is not None:
                mask['point_coords'] = np.array(mask['point_coords']) * np.array([scale_w, scale_h])
            
        return masks
    
    def segment_image(
        self, 
        image: Union[np.ndarray, Image.Image],
        mask_format: str = 'rle'
    ) -> List[Dict[str, Any]]:
        """
        Segment an image and generate masks.
        
        Args:
            image: Input image as numpy array (preferred) or PIL Image.
                   Will be converted to numpy array and RGB format if needed.
            
        Returns:
            List of dictionaries containing mask information, where each dictionary has:
            - segmentation: Binary mask array
            - area: Area of the mask in pixels
            - bbox: Bounding box in [x, y, width, height] format
            - predicted_iou: Model's prediction of mask quality
            - point_coords: Sampled input point coordinates
            - stability_score: Additional measure of mask quality
            - crop_box: Crop box used for generation in [x, y, width, height] format
        """
        # Convert to numpy array and ensure RGB format
        image = convert_image_format(image, target_format="numpy", ensure_rgb=True)
        
        # Downsample image for segmentation
        downsampled_image, original_size = self._downsample_image(image)
            
        # Generate masks using SAM2
        masks = self.mask_generator.generate(downsampled_image)
        
        # Scale masks back to original size
        masks = self._upscale_masks(masks, original_size)
        
        # Convert masks to requested format
        converted_masks = []
        for mask_data in masks:
            binary_mask = mask_data["segmentation"]
            converted_mask = convert_mask_format(binary_mask, target_format=mask_format)
            mask_data['segmentation'] = converted_mask
            converted_masks.append(mask_data)

        return converted_masks

    def _generate_mask_hash(self, image_hash: str, mask_data: Dict[str, Any]) -> str:
        """
        Generate a unique hash for a mask based on image hash and mask properties.
        
        Args:
            image_hash: Hash of the parent image
            mask_data: Dictionary containing mask information
            
        Returns:
            str: Unique hash for the mask
        """
        # Create a string combining relevant mask properties
        mask_string = (
            f"{image_hash}"
            f"{str(mask_data['segmentation'])}"
            f"{str(mask_data['bbox'])}"
        )
        
        # Generate MD5 hash
        return hashlib.md5(mask_string.encode()).hexdigest()

    def _convert_mask_for_mongodb(self, mask_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert mask data to MongoDB-compatible format.
        Only converts NumPy arrays that aren't already in RLE format.
        
        Args:
            mask_data: Dictionary containing mask information
            
        Returns:
            Dictionary with MongoDB-compatible data types
        """
        converted_mask = mask_data.copy()
        
        # Convert only point_coords if it exists (segmentation is already RLE)
        if 'point_coords' in converted_mask and converted_mask['point_coords'] is not None:
            converted_mask['point_coords'] = converted_mask['point_coords'].tolist()
            
        return converted_mask

    def save_segmentation_metadata(
        self,
        image_hash: str,
        masks: List[Dict[str, Any]],
        mongodb_manager: MongoDBManager,
    ) -> bool:
        """
        Save segmentation metadata to MongoDB.
        
        Args:
            image_hash: Hash of the image in MongoDB.
            masks: List of masks to save.
            mongodb_manager: MongoDBManager instance to access the database.
            
        Returns:
            bool: True if the document was updated, False otherwise.
        """
        # Convert masks to MongoDB-compatible format
        converted_masks = []
        for mask in masks:
            # Add unique hash to mask
            mask['mask_hash'] = self._generate_mask_hash(image_hash, mask)
            # Convert mask data
            converted_mask = self._convert_mask_for_mongodb(mask)
            converted_masks.append(converted_mask)
        
        # Use the MongoDBManager's update_one method to overwrite existing masks
        return mongodb_manager.update_one(
            collection_name=mongodb_manager.mongodb_config.image_metadata_collection,
            filter_query={"image_hash": image_hash},
            update_data={"masks": converted_masks},
            upsert=False
        )