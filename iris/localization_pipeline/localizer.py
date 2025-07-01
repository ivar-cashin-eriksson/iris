import torch
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO
from transformers import AutoProcessor, YolosForObjectDetection, YolosImageProcessor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from abc import ABC, abstractmethod

from iris.config.localization_pipeline_config_manager import (
    LocalizationModelConfig, 
    SAM2Config, 
    YoloConfig, 
    YolosConfig
)
from iris.mixins.renderable import RenderableMixin
from iris.protocols.context_protocols import HasImageContext
from iris.utils.machine_utils import get_device
from iris.utils.image_utils import convert_mask_format, convert_image_format


class LocalizationModel(ABC):
    """Base class for object localization models."""

    @abstractmethod
    def localize_objects(
        self, 
        image: np.ndarray
    ) -> list[dict[str, any]]:
        """
        Perform object localization on an image.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            list[dict[str, any]]: List of dictionaries containing localization 
                                  information in relative coordinates.
        """
        ...

    def post_process_masks(
        self, 
        masks: list[dict[str, any]]
    ) -> list[dict[str, any]]:
        """
        Post-process masks to ensure they are valuable for downstream tasks.
        
        Args:
            masks: List of mask dictionaries.
            
        Returns:
            List of post-processed mask dictionaries.
        """
        # Placeholder for post-processing logic
        return masks

    def _to_relative_coordinates(
        self, 
        localization_data: dict[str, any], 
        shape: tuple[int, ...]
    ) -> dict[str, any]:
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


class BoundingBoxModel(LocalizationModel):
    """Base class for models that only produce bounding boxes."""
    

class YoloModel(BoundingBoxModel):
    """YOlO model for object localization."""

    def __init__(self, model_config: YoloConfig):
        """
        Initialize the YOLO model with configuration parameters.
        
        Args:
            model_config (YolosConfig): Configuration object containing all parameters
        """
        self.model_config = model_config
        
        # Initialize model and processor from local checkpoint
        self.model = YOLO("../checkpoints/detect/train/weights/best.pt")

    def localize_objects(
        self, 
        image: np.ndarray
    ) -> list[dict[str, any]]:
        """
        Perform object localization on an image.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            list[dict[str, any]]: List of dictionaries containing localization 
                                  information in relative coordinates.
        """

        # Run prediction (YOLO expects BGR format)
        results = self.model.predict(
            source=image[..., ::-1],  # Convert to BGR format
            **self.model_config.model_params
        )
        boxes = results[0].boxes

        # Convert detections to standard format with relative coordinates
        detections = []
        for score, label, box in zip(boxes.conf, boxes.cls, boxes.xyxyn):
            box = box.cpu().tolist()
            detection = {
                'label': results[0].names[label.item()],
                'label_id': label.item(),
                'score': score.item(),
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            }
            
            detections.append(detection)

        return detections


class YolosModel(BoundingBoxModel):
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
    ) -> list[dict[str, any]]:
        """
        Perform object localization on an image.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            list[dict[str, any]]: List of dictionaries containing localization 
                                  information in relative coordinates.
        """
        # Convert numpy array to PIL Image for YOLOS processor
        pil_image = PILImage.fromarray(image)
    
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
                'label_id': label.item(),
                'score': score.item(),
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            }
            
            # Convert to relative coordinates
            detection = self._to_relative_coordinates(
                detection, image.shape[:2]
            )
            detections.append(detection)

        return detections


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

    def localize_objects(
        self, 
        image: np.ndarray
    ) -> list[dict[str, any]]:
        """
        Perform object localization on an image.
        
        Args:
            image: Input image as numpy array in RGB format.
                        
        Returns:
            list[dict[str, any]]: List of dictionaries containing localization 
                                  information in relative coordinates.
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
        localization_data: dict[str, any], 
        shape: tuple[int, ...]
    ) -> dict[str, any]:
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


class Localizer:
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
            case YoloConfig():
                self.model: LocalizationModel = YoloModel(self.model_config)
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
        pil_img = PILImage.fromarray(image)
        resized_img = pil_img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
        return np.array(resized_img), original_size
    
    def localize(
        self, 
        image: RenderableMixin,
        context: HasImageContext,
        post_process: bool = True,
    ) -> list[dict[str, any]]:
        """
        Object localize an image and generate bounding boxes.
        
        Args:
            image (RenderableMixin): Input image as RenderableMixin. Will be 
                                     converted to numpy array in RGB format.
            context (HasImageContext): Context providing PILImage.
            
        Returns:
            List of dictionaries containing localization information, where 
            each dictionary has at least:
            - label: Name of the detected object
            - label_id: ID of the detected object
            - score: Confidence score of the detection
            - bbox: Bounding box in relative [x, y, width, height] format
        """
        # Always convert to numpy array first thing
        np_image = convert_image_format(
            image.render(context), 
            target_format="numpy", 
            ensure_rgb=True
        )
        
        # Downsample image for segmentation
        downsampled_image, _ = self._downsample_image(np_image)

        localizations = self.model.localize_objects(downsampled_image)

        if post_process:
            localizations = self.model.post_process_masks(localizations)

        return localizations
