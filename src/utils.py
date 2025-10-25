"""
Common utility functions for CLIP and Pre-CLIP analysis.

This module provides shared functions for:
- Loading the F1 dataset (images and captions)
- Computing cosine similarity between embeddings
- Saving results and embeddings for analysis
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image


def load_dataset(dataset_path):
    """
    Load images and captions from the F1 dataset.

    Args:
        dataset_path: Path to dataset root directory containing captions.json

    Returns:
        Tuple of (images, captions, metadata) where:
            - images: List of PIL Image objects
            - captions: List of caption strings
            - metadata: List of dicts with image info (id, category, filename)
    """
    captions_file = os.path.join(dataset_path, "captions.json")

    with open(captions_file, 'r') as f:
        data = json.load(f)

    images = []
    captions = []
    metadata = []

    for item in data['images']:
        img_path = os.path.join(dataset_path, item['filename'])
        img = Image.open(img_path).convert('RGB')
        images.append(img)
        captions.append(item['caption'])
        metadata.append({
            'id': item['id'],
            'category': item['category'],
            'filename': item['filename']
        })

    return images, captions, metadata


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings1: Tensor of shape (n, dim)
        embeddings2: Tensor of shape (m, dim)

    Returns:
        Similarity matrix of shape (n, m)
    """
    # Normalize embeddings
    embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())

    return similarity


def save_embeddings(embeddings, metadata, output_path, prefix):
    """
    Save embeddings and metadata to disk.

    Args:
        embeddings: Tensor of embeddings
        metadata: List of metadata dicts
        output_path: Directory to save files
        prefix: Prefix for filenames (e.g., 'image' or 'text')
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert to numpy and save
    embeddings_np = embeddings.cpu().numpy()
    np.save(
        os.path.join(output_path, f'{prefix}_embeddings.npy'),
        embeddings_np
    )

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(
        os.path.join(output_path, f'{prefix}_metadata.csv'),
        index=False
    )


def save_similarity_matrix(similarity_matrix, metadata, output_path, filename):
    """
    Save similarity matrix as CSV with row/column labels.

    Args:
        similarity_matrix: Tensor of similarity scores
        metadata: List of metadata dicts for labeling
        output_path: Directory to save file
        filename: Name of output CSV file
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert to numpy
    similarity_np = similarity_matrix.cpu().numpy()

    # Create labels from metadata
    labels = [f"{m['category']}_{m['id']}" for m in metadata]

    # Create DataFrame
    df = pd.DataFrame(
        similarity_np,
        index=labels,
        columns=labels
    )

    # Save
    df.to_csv(os.path.join(output_path, filename))
