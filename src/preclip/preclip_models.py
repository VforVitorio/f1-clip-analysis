"""
Pre-CLIP image and text encoders for embedding extraction.

This module implements separate pre-trained models for image and text encoding. 
This represents the "pre-CLIP" approach where vision and language models were trained independently
rather than jointly.

Components:
    - ImageEncoder: ResNet50 (pretrained on ImageNet) for extracting visual features
    - TextEncoder: DistilUSE multilingual model for extracting text features

Both encoders auto-detect and use the GPU if available, apply appropriate preprocessing and normalization
support both single samples and batch processing and return L2-normalized embeddings as torch.Tensors

The embeddings from these models will be compared using cosine similarity to analyze the matching 
performance before the multimodal CLIP approach
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from PIL import Image


class ImageEncoder:
    """
    ResNet50-based image encoder for extracting visual features.

    Uses pretrained ResNet50 without the classification head, extracting 2048-dimensional
    feature vectors from the final pooling layer.
    """

    def __init__(self):
        """
        Initialize ResNet50 encoder and preprocessing pipeline
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        # Remove classification head
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._embedding_dim = 2048

    def get_embedding_dim(self):
        """Return the dimensionality of the embeddings."""
        return self._embedding_dim

    def encode(self, images):
        """
        Extract and normalize embeddings from images.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            L2-normalized embeddings as torch.Tensor of shape (n, 2048)
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]

        # Preprocess images
        image_tensors = torch.stack([self.preprocess(img) for img in images])
        image_tensors = image_tensors.to(self.device)

        # Extract features
        with torch.no_grad():
            embeddings = self.model(image_tensors)
            # Remove spatial dimensions
            embeddings = embeddings.squeeze(-1).squeeze(-1)

        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


class TextEncoder:
    """
    Sentence-transformer based text encoder for extracting semantic features.

    Uses distiluse-base-multilingual-cased model to generate 512-dimensional
    sentence embeddings.
    """

    def __init__(self):
        """
        Initialize sentence-transformer model
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load sentence-transformer model
        self.model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased')
        self.model.to(self.device)

        self._embedding_dim = 512

    def get_embedding_dim(self):
        """Return the dimensionality of the embeddings."""
        return self._embedding_dim

    def encode(self, texts):
        """
        Extract and normalize embeddings from text.

        Args:
            texts: Single string or list of strings

        Returns:
            L2-normalized embeddings as torch.Tensor of shape (n, 512)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Extract embeddings (already normalized by sentence-transformers)
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        )

        return embeddings
