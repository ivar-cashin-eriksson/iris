"""
Handler for image segmentation using SAM2 model.
"""

import torch
import numpy as np
from PIL import Image
from typing import Any
from transformers import AutoProcessor, YolosForObjectDetection, YolosImageProcessor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from abc import ABC, abstractmethod
import hashlib

from iris.config.localization_pipeline_config_manager import LocalizationModelConfig, SAM2Config, YolosConfig
from iris.utils.machine_utils import get_device
from iris.utils.image_utils import convert_mask_format, convert_image_format
from iris.data_pipeline.mongodb_manager import MongoDBManager


class LocalizationModel(ABC):
    """Base class for object localization models."""

    @abstractmethod
    def localize_objects(
        self, 
        image: np.ndarray  # Only accept numpy array
    ) -> list[dict[str, Any]]:
        """
        Perform object localization on an image.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            List of dictionaries containing mask information.
        """
        pass

    def post_process_masks(
        self, 
        masks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Post-process masks to ensure they are valuable for downstream tasks.
        
        Args:
            masks: List of mask dictionaries.
            
        Returns:
            List of post-processed mask dictionaries.
        """
        # Placeholder for post-processing logic
        return masks
    
    @abstractmethod
    def generate_localization_hash(
        self, 
        image_hash: str, 
        localization_data: dict[str, Any]
    ) -> str:
        """
        Generate a unique hash for an object localization based on image hash and mask properties.
        
        Args:
            image_hash (str): Hash of the parent image
            localization_data (dict): Dictionary containing object localization information
            
        Returns:
            str: Unique hash for the object localization
        """
        pass

    def _to_relative_coordinates(
        self, 
        localization_data: dict[str, Any], 
        shape: tuple[int, ...]
    ) -> dict[str, Any]:
        """
        Convert coordinates to relative format.
        
        Args:
            localization_data: Localization data containing coordinates
            shape: Image shape as (height, width, channels) or (height, width)
            
        Returns:
            dict: Localization data with relative coordinates
        """        
        height, width = shape[:2]  # Extract height and width from shape
        result = localization_data.copy()
        
        # Convert bbox [x,y,w,h]
        bbox = result['bbox']
        result['bbox'] = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]

        return result


class YolosModel(LocalizationModel):
    """YOLOS model for object localization."""

    def __init__(self, model_config: YolosConfig):
        """
        Initialize the YOLOS model with configuration parameters.
        
        Args:
            model_config (YolosConfig): Configuration object containing all parameters
        """
        self.model_config = model_config
        
        # Initialize model and processor from local checkpoint
        self.model: YolosForObjectDetection = YolosForObjectDetection.from_pretrained(
            self.model_config.get_checkpoint_path()
        )
        self.processor: YolosImageProcessor = AutoProcessor.from_pretrained(
            self.model_config.get_checkpoint_path()
        )
        self.model.to(get_device(self.model_config.device))

    def localize_objects(
            self, 
            image: np.ndarray
        ) -> list[dict[str, Any]]:
        """
        Segment an image using the YOLOS model.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            List of dictionaries containing mask information.
        """
        # Convert numpy array to PIL Image for YOLOS processor
        pil_image = Image.fromarray(image)
    
        # Preprocess and run inference
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(get_device(self.model_config.device)) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        # Postprocess results
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.model_config.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Convert detections to standard format with relative coordinates
        detections = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            box = box.cpu().tolist()
            detection = {
                'label': self.model.config.id2label[label.item()],
                'score': score.item(),
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            }
            
            # Convert to relative coordinates
            detection = self._to_relative_coordinates(
                detection, image.shape[:2]
            )
            detections.append(detection)

        return detections

    def generate_localization_hash(self, image_hash, localization_data):
        """
        Generate a unique hash for an object localization based on parent image
        hash and bounding box properties.
        
        Args:
            image_hash (str): Hash of the parent image
            localization_data (dict): Dictionary containing object localization
                                      information
            
        Returns:
            str: Unique hash for the object localization
        """
        # Create a string combining relevant mask properties
        mask_string = (
            f"{image_hash}"
            f"{str(localization_data['bbox'])}"
        )
        
        # Generate MD5 hash
        return hashlib.md5(mask_string.encode()).hexdigest()


class SAM2Model(LocalizationModel):
    """SAM2 model for object localization."""

    def __init__(self, model_config: SAM2Config):
        """
        Initialize the SAM2 model with configuration parameters.
        
        Args:
            model_config: SAM2Config object containing all parameters
        """
        self.model_config = model_config

        # Initialize SAM2 model
        sam2 = sam_model_registry[
            self.model_config.model_type
        ](checkpoint=self.model_config.get_checkpoint_path())
        sam2.to(get_device(self.model_config.device))
        sam2.eval()
        
        # Initialize mask generator with configuration parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam2,
            **self.model_config.model_params
        )

    def generate_localization_hash(self, image_hash: str, localization_data: dict[str, Any]) -> str:
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
            f"{str(localization_data['segmentation'])}"
            f"{str(localization_data['bbox'])}"
        )
        
        # Generate MD5 hash
        return hashlib.md5(mask_string.encode()).hexdigest()

    def localize_objects(
            self, 
            image: np.ndarray
        ) -> list[dict[str, Any]]:
        """
        Segment an image using the SAM2 model.
        
        Args:
            image (np.ndarray): Input image.
                        
        Returns:
            List of dictionaries containing mask information.
        """
        masks = self.mask_generator.generate(image)

        # Convert masks to requested format
        converted_masks = []
        for mask_data in masks:
            # Convert mask format first
            binary_mask = mask_data["segmentation"]
            mask_data['segmentation'] = convert_mask_format(
                binary_mask, 
                target_format=self.model_config.mask_format
            )
            
            # Convert coordinates to relative
            mask_data = self._to_relative_coordinates(
                mask_data, image.shape
            )
            converted_masks.append(mask_data)

        return converted_masks

    def _to_relative_coordinates(
        self, 
        localization_data: dict[str, Any], 
        shape: tuple[int, ...]
    ) -> dict[str, Any]:
        """Convert SAM2 coordinates to relative format."""
        height, width = shape[:2]
        result = super()._to_relative_coordinates(
            localization_data, shape
        )
        
        # Convert area
        result['area'] = result['area'] / (width * height)
            
        # Convert point_coords if present [[x,y]]
        if 'point_coords' in result and result['point_coords'] is not None:
            result['point_coords'] = [
                [x / width, y / height] 
                for [x, y] in result['point_coords']
            ]
            
        # Convert crop_box if present [x1,y1,x2,y2]
        if 'crop_box' in result and result['crop_box'] is not None:
            crop_box = result['crop_box']
            result['crop_box'] = [
                crop_box[0] / width,
                crop_box[1] / height,
                crop_box[2] / width,
                crop_box[3] / height
            ]
            
        return result


class LocalizationHandler:
    """Handles image segmentation using SAM2 model."""
    
    def __init__(self, model_config: LocalizationModelConfig):
        """
        Initialize the SegmentationHandler.
        
        Args:
            sam2_config: SegmentationConfig object containing all 
                         configuration parameters
        """
        self.model_config = model_config
        
        # Configure CUDA settings if available
        if self.model_config.device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Initialize Object Localization Model
        match self.model_config:
            case YolosConfig():
                self.model: LocalizationModel = YolosModel(self.model_config)
            case SAM2Config():
                self.model: LocalizationModel = SAM2Model(self.model_config)
            case _:
                raise ValueError(f"Unsupported model config type: {type(self.model_config)}")
        
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
        
        # If scaling is deactivated, return original image and size
        if self.model_config.max_image_size is None:
            return image, original_size
        
        # Calculate scaling factor
        scale = self.model_config.max_image_size / max(height, width)
        if scale >= 1:  # Don't upscale
            return image, original_size
            
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Use PIL for high-quality downsampling
        pil_img = Image.fromarray(image)
        resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(resized_img), original_size
    
    def localize_image(
        self, 
        image: np.ndarray | Image.Image,
        post_process: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Object localize an image and generate bounding boxes.
        
        Args:
            image: Input image as numpy array or PIL Image.
                   Will be converted to numpy array and RGB format.
            
        Returns:
            List of dictionaries containing localization information, where 
            each dictionary has at least:
            - bbox: Bounding box in [x, y, width, height] format
        """
        # Always convert to numpy array first thing
        image = convert_image_format(image, target_format="numpy", ensure_rgb=True)
        
        # Downsample image for segmentation
        downsampled_image, _ = self._downsample_image(image)

        masks = self.model.localize_objects(downsampled_image)

        if post_process:
            masks = self.model.post_process_masks(masks)

        return masks            

    def save_segmentation_metadata(
        self,
        image_hash: str,
        localizations: list[dict[str, Any]],
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
        updated_localizations = []
        for localization in localizations:
            # Add unique hash to mask
            localization['localization_hash'] = self.model.generate_localization_hash(
                image_hash, localization
            )

            # Add model type info to localization
            match self.model:
                case YolosModel():
                    model_type = "yolos"
                case SAM2Model():
                    model_type = "sam2"
                case _:
                    raise ValueError(f"Unsupported model type: {type(self.model)}")

            localization['model_type'] = model_type

            updated_localizations.append(localization)
        

        # Use the MongoDBManager's update_one method to overwrite existing localizations
        return mongodb_manager.update_one(
            collection_name=mongodb_manager.mongodb_config.image_metadata_collection,
            filter_query={"image_hash": image_hash},
            update_data={"localizations": updated_localizations},
            upsert=False
        )