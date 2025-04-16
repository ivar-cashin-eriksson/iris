import open_clip
import torch
import torchvision.transforms as T
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

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Generate embeddings for a single PIL image.
        
        Args:
            image: A PIL Image
            
        Returns:
            torch.Tensor: Image embeddings
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        
        processed_image = self.preprocess(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu()[0]  # Return as tensor, removing batch dimension

    def get_augmented_embedding(self, img: Image.Image, num_augmentations: int = 5) -> torch.Tensor:
        """
        Generate embeddings by averaging original and augmented versions of the image.
        
        Uses data augmentation techniques including random affine transformations and
        horizontal flips to create multiple versions of the image. The embeddings
        for all versions are averaged to create a more robust representation.

        Args:
            img (Image.Image): A PIL Image to generate embeddings for
            num_augmentations (int, optional): Number of augmented versions to create. Defaults to 5.
            
        Returns:
            torch.Tensor: Averaged embedding tensor for all image versions
            
        Raises:
            ValueError: If input is not a PIL Image
        """
        # Define augmentations
        augmentation = T.Compose([
            T.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=15
            ),
            T.RandomHorizontalFlip(p=0.5),
        ])
        
        # Convert PIL image to tensor for transformations
        to_tensor = T.ToTensor()
        to_pil = T.ToPILImage()
        img_tensor = to_tensor(img)
        
        # Get embeddings for original and augmented versions
        embeddings = [self.get_embedding(img)]
        
        # Get augmented versions
        for _ in range(num_augmentations):
            # Apply augmentation
            aug_tensor = augmentation(img_tensor)
            aug_image = to_pil(aug_tensor)
            
            # Get embedding
            aug_embedding = self.get_embedding(aug_image)
            embeddings.append(aug_embedding)
        
        # Stack and average all embeddings
        stacked = torch.stack(embeddings)
        return stacked.mean(dim=0)
