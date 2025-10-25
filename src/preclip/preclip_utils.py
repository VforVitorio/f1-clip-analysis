"""
Utility functions specific to Pre-CLIP analysis.

Pre-CLIP uses separate image (ResNet50) and text (sentence-transformers) encoders
that produce embeddings with different dimensions. This module provides functions
to handle this dimension mismatch using PCA projection.
"""

import sys
from pathlib import Path
import torch
from sklearn.decomposition import PCA

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import compute_cosine_similarity as _base_compute_cosine_similarity


def project_to_common_space(embeddings1, embeddings2):
    """
    Project embeddings to common dimensionality using PCA.

    This is needed for Pre-CLIP because image and text encoders
    produce embeddings with different dimensions.

    Args:
        embeddings1: Tensor of shape (n, dim1)
        embeddings2: Tensor of shape (m, dim2)

    Returns:
        Tuple of projected tensors with same dimensionality
    """
    emb1_np = embeddings1.cpu().numpy()
    emb2_np = embeddings2.cpu().numpy()

    # Use minimum dimension between both embeddings and n_samples
    n_samples = min(emb1_np.shape[0], emb2_np.shape[0])
    target_dim = min(emb1_np.shape[1], emb2_np.shape[1], n_samples - 1)

    # Project first embeddings if needed
    if emb1_np.shape[1] != target_dim:
        pca1 = PCA(n_components=target_dim)
        emb1_np = pca1.fit_transform(emb1_np)

    # Project second embeddings if needed
    if emb2_np.shape[1] != target_dim:
        pca2 = PCA(n_components=target_dim)
        emb2_np = pca2.fit_transform(emb2_np)

    return torch.from_numpy(emb1_np).float(), torch.from_numpy(emb2_np).float()


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity matrix between two sets of embeddings.

    This Pre-CLIP version handles embeddings with different dimensions
    by projecting them to a common space using PCA before computing similarity.

    Args:
        embeddings1: Tensor of shape (n, dim1)
        embeddings2: Tensor of shape (m, dim2)

    Returns:
        Similarity matrix of shape (n, m)
    """
    # Project to common space if dimensions differ
    if embeddings1.shape[1] != embeddings2.shape[1]:
        embeddings1, embeddings2 = project_to_common_space(
            embeddings1, embeddings2)

    # Use base implementation from utils
    return _base_compute_cosine_similarity(embeddings1, embeddings2)
