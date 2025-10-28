"""
CLIP multimodal encoder for joint image-text embeddings.

Uses OpenAI's CLIP (ViT-B/32) to produce aligned 512-dim embeddings
in a shared space for both images and text.
"""


import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class CLIPEncoder:
    """
    Unified CLIP encoder for both images and text.

    Uses ViT-B/32 model to generate 512-dim embeddings in shared space.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor."""

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim = 512

    def get_embedding_dim(self):
        """Return embedding dimensionality."""
        return self._embedding_dim

    def encode_images(self, images):
        """Extract normalized image embeddings."""

        if isinstance(images, Image.Image):
            images = [images]

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        # Move inputs to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # Extract features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize
        image_features = torch.nn.functional.normalize(
            image_features, p=2, dim=1)

        return image_features

    def encode_texts(self, texts):
        """Extract normalized text embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        # Process texts
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)

        # Move inputs to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # Extract features
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize
        text_features = torch.nn.functional.normalize(
            text_features, p=2, dim=1)

        return text_features
