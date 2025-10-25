"""
CLIP multimodal encoder for joint image-text embeddings.

This module implements CLIP (Contrastive Language-Image Pre-training),
a neural network trained on image-text pairs that produces aligned embeddings
in a shared 512-dimensional space.


The main file component is the CLIPEncoder class, an unified encoder for both images and text

The encoder auto-detects GPU availability and returns L2-normalized embeddings that enable direct comparison
between visual and textual content.

"""


import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class CLIPEncoder:
    """
    CLIP-based encoder for joint image-text embedding extraction.

    Uses OpenAI´s CLIP ViT-B/32 model to generate 512-dimensional embeddings 
    in a shared multimodal space

    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model and processor

        model_name: HuggingFace model identifier
        """

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
        """
        Returns the dimensionality of embeddings
        """
        return self._embedding_dim

    def encode_images(self, images):
        """
        Extract and normalize image embeddings.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            L2-normalized embeddings as torch.Tensor of shape (n, 512)
        """

        if isinstance(images, Image.Image):
            images = [images]

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        # Move inputs to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # Extract feautes
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # L2 normalization
        image_features = torch.nn.functional.normalize(
            image_features, p=2, dim=1)

        return image_features

    def encode_texts(self, texts):
        """
        Extract and normalize text embeddings.

        Args:
            texts: Single string or list of strings

        Returns:
            L2-normalized embeddings as torch.Tensor of shape (n, 512)
        """
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

        # L2 normalization
        text_features = torch.nn.functional.normalize(
            text_features, p=2, dim=1)

        return text_features
