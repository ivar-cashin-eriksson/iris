import open_clip
import torch
from PIL import Image
import numpy as np

from iris.config.embedding_pipeline_config_manager import ClipConfig

class EmbeddingHandler:
    def __init__(self, config: ClipConfig):
        """
        Initialize the EmbeddingHandler with specified CLIP model.
        
        Args:
            config (ClipConfig): Clip configuration instance
        """
        self.config = config
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.config.device
        )
        self.model.eval()

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings for a single PIL image.
        
        Args:
            image: A PIL Image
            
        Returns:
            numpy.ndarray: Image embeddings
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        
        processed_image = self.preprocess(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
