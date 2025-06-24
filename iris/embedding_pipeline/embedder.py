import open_clip
import torch
from PIL import Image as PILImage

from iris.config.embedding_pipeline_config_manager import ClipConfig
from iris.mixins.embeddable import EmbeddingPayload, EmbeddableMixin
from iris.protocols.context_protocols import HasFullContext
from iris.utils.log import logger


class Embedder:
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

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a text string using the CLIP model.

        Args:
            text (str): The text to embed.

        Returns:
            torch.Tensor: The embedding vector for the text.
        """
        with torch.no_grad():
            text_tokens = open_clip.tokenize([text]).to(self.config.device)
            text_embedding = self.model.encode_text(text_tokens).squeeze(0) # TODO: Do not remove batch dimension 
            return text_embedding
        
    def embed_image(self, img: PILImage) -> torch.Tensor:
        """
        Embed a PIL image using the CLIP model.

        Args:
            img (PILImage): The image to embed.

        Returns:
            torch.Tensor: The embedding vector for the image.
        """
        if not isinstance(img, PILImage.Image):
            raise ValueError("Input must be a PIL Image.")
        
        processed_image = self.preprocess(img).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu()[0]
        
    def embed(
        self, 
        item: EmbeddingPayload | EmbeddableMixin,
        context: HasFullContext | None = None,
    ) -> torch.Tensor:
        """
        Retrieve or compute an embedding for this document.

        The method first checks for a cached embedding in memory, then tries to fetch
        from Qdrant. If no embedding is found, it falls back to computing it using the
        provided embedder.

        Args:
            qdrant_manager (QdrantManager | None): Optional manager to fetch from Qdrant.
            embedder (Embedder | None): Optional embedder to compute the embedding if needed.

        Returns:
            torch.Tensor: The final embedding.

        Raises:
            RuntimeError: If the embedding could not be retrieved or computed.
        """
        match item:
            case EmbeddingPayload():
                # If the item is an EmbeddingPayload, we can directly use its components
                embedding_data = item
            case EmbeddableMixin():
                # If the item is an EmbeddableMixin, we need to get its embedding data
                if context is None:
                    raise ValueError("Context must be provided for EmbeddableMixin items.")
                embedding_data = item.get_embedding_data(context)
            case _:
                raise TypeError(f"Unsupported item type: {type(item)}")

        # average the embeddings of all components
        embeddings = []
        for component in embedding_data.components:
            match component.type:
                case "text":
                    embeddings.append(self.embed_text(component.content))
                case "image":
                    embeddings.append(self.embed_image(component.content))
            
        if not embeddings:
            raise RuntimeError("No embeddings could be computed for the provided item.")
        
        stacked_embeddings = torch.stack(embeddings)
        final_embedding = torch.mean(stacked_embeddings, dim=0)
            
        logger.debug(f"Computed embedding: {final_embedding}")
        return final_embedding
    
    def embed_batch(
        self,
        items: list[EmbeddableMixin],
        context: HasFullContext,
    ) -> list[torch.Tensor]:
        """
        Batch process multiple EmbeddableMixin items and return their embeddings.
        This implementation properly batches text and image components for efficiency.

        Args:
            items (list[EmbeddableMixin]): List of items to embed
            context (HasFullContext): Context for retrieving embedding data

        Returns:
            list[torch.Tensor]: List of embeddings for each item
        """
        # Get embedding data for all items first
        all_embedding_data = [item.get_embedding_data(context) for item in items]
        
        # Collect all text and image components
        text_batches = []
        image_batches = []
        component_maps = []  # Track which components belong to which item
        
        for item_idx, embedding_data in enumerate(all_embedding_data):
            text_components = []
            image_components = []
            for comp in embedding_data.components:
                match comp.type:
                    case "text":
                        text_components.append(comp.content)
                    case "image":
                        image_components.append(comp.content)
                    case _:
                        raise TypeError(f"Unsupported component type: {comp.type}")
            
            if text_components:
                text_batches.extend(text_components)
            if image_components:
                image_batches.extend(image_components)
            component_maps.append({
                'text_start': len(text_batches) - len(text_components),
                'text_count': len(text_components),
                'image_start': len(image_batches) - len(image_components),
                'image_count': len(image_components)
            })

        # Process all texts in one batch
        text_embeddings = []
        if text_batches:
            with torch.no_grad():
                text_tokens = open_clip.tokenize(text_batches).to(self.config.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_embeddings = text_features.cpu()

        # Process all images in one batch
        image_embeddings = []
        if image_batches:
            processed_images = torch.stack([
                self.preprocess(img) for img in image_batches
            ]).to(self.config.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(processed_images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embeddings = image_features.cpu()

        # Reconstruct individual item embeddings
        final_embeddings = []
        for item_map in component_maps:
            item_embeddings = []
            
            # Add text embeddings for this item
            if item_map['text_count'] > 0:
                start = item_map['text_start']
                end = start + item_map['text_count']
                item_embeddings.append(text_embeddings[start:end])
            
            # Add image embeddings for this item
            if item_map['image_count'] > 0:
                start = item_map['image_start']
                end = start + item_map['image_count']
                item_embeddings.append(image_embeddings[start:end])
            
            # Average all embeddings for this item
            if item_embeddings:
                item_embeddings = torch.cat(item_embeddings)
                final_embeddings.append(torch.mean(item_embeddings, dim=0))
            else:
                raise RuntimeError("No embeddings could be computed for an item.")
        
        return final_embeddings