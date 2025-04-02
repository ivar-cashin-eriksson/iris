"""
Handler for image segmentation using SAM2 model.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from iris.config.segmentation_pipeline_config_manager import SAM2Config
from iris.segmentation_pipeline.utils import get_device
from iris.utils import convert_image_format
from iris.segmentation_pipeline.utils import convert_mask_format
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
            
        # Generate masks using SAM2
        masks = self.mask_generator.generate(image)
        
        # TODO: Filter masks to only include clothing items
        
        converted_masks = []
        for mask_data in masks:
            # Convert the mask to the requested format
            binary_mask = mask_data["segmentation"]
            converted_mask = convert_mask_format(binary_mask, target_format=mask_format)
            
            mask_data['segmentation'] = converted_mask
            
            converted_masks.append(mask_data)

        return converted_masks

    
    def save_segmentation_metadata(
        self,
        image_hash: str,
        masks: List[Dict[str, Any]],
        mongodb_manager: MongoDBManager,  # Pass the MongoDBManager instance
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
        # Use the MongoDBManager's update_one method to overwrite existing masks
        return mongodb_manager.update_one(
            collection_name=mongodb_manager.mongodb_config.image_metadata_collection,
            filter_query={"image_hash": image_hash},
            update_data={"masks": masks},
            upsert=False
        ) 