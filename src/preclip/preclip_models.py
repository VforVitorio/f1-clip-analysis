"""
Pre-CLIP image and text encoders.

Uses separate pretrained models:
- ResNet50 (ImageNet) for images → 2048-dim features
- DistilUSE multilingual for text → 512-dim features

Both models auto-detect GPU and return normalized embeddings.
This represents the old approach before joint training.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from PIL import Image


class ImageEncoder:
    """
    ResNet50 image encoder (pretrained on ImageNet).

    Extracts 2048-dim features from the final pooling layer.
    """

    def __init__(self):
        """Initialize ResNet50 and preprocessing."""
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
        """Return embedding dimensionality."""
        return self._embedding_dim

    def encode(self, images):
        """Extract normalized embeddings from images."""
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
    Sentence-transformer text encoder.

    Uses distiluse-base-multilingual-cased to generate 512-dim embeddings.
    """

    def __init__(self):
        """Initialize sentence-transformer model."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load sentence-transformer model
        self.model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased')
        self.model.to(self.device)

        self._embedding_dim = 512

    def get_embedding_dim(self):
        """Return embedding dimensionality."""
        return self._embedding_dim

    def encode(self, texts):
        """Extract normalized embeddings from text."""
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
