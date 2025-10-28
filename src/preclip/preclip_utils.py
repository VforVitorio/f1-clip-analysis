"""
Pre-CLIP utilities for handling different embedding dimensions.

Uses PCA to project image and text embeddings to a common space
since ResNet50 and sentence-transformers output different dimensions.
"""

from utils import compute_cosine_similarity as _base_compute_cosine_similarity
import sys
from pathlib import Path
import torch
from sklearn.decomposition import PCA

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def project_to_common_space(embeddings1, embeddings2):
    """
    Project embeddings to common dimensionality using PCA.

    Needed because image and text encoders produce different dimensions.
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
    Compute cosine similarity between embeddings with different dimensions.

    Projects to common space via PCA if needed before computing similarity.
    """
    # Project to common space if dimensions differ
    if embeddings1.shape[1] != embeddings2.shape[1]:
        embeddings1, embeddings2 = project_to_common_space(
            embeddings1, embeddings2)

    # Use base implementation from utils
    return _base_compute_cosine_similarity(embeddings1, embeddings2)
