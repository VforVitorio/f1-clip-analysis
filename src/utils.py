"""
Shared utilities for CLIP and Pre-CLIP analysis.

Handles dataset loading, similarity computation, and saving results.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image


def load_dataset(dataset_path):
    """
    Load images and captions from dataset directory.

    Returns tuple of (images, captions, metadata).
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
    """Compute cosine similarity between two embedding sets."""
    # Normalize embeddings
    embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())

    return similarity


def save_embeddings(embeddings, metadata, output_path, prefix):
    """Save embeddings and metadata to disk as .npy and .csv files."""
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
    """Save similarity matrix as CSV with labels from metadata."""
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
