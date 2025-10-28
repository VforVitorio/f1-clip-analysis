"""
CLIP analysis script.

Extracts embeddings using unified CLIP model for both images and text,
computes similarity in shared embedding space, and saves results.
"""

from utils import (
    load_dataset,
    compute_cosine_similarity,
    save_embeddings,
    save_similarity_matrix
)
from clip_models import CLIPEncoder
import sys
import torch
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_paths():
    """Set up dataset and output directories."""
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = project_root / "dataset"
    output_path = project_root / "results" / "clip"

    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")

    return dataset_path, output_path


def initialize_encoder():
    """Initialize CLIP encoder."""
    print("\nInitializing CLIP encoder...")
    encoder = CLIPEncoder()

    print(f"Device: {encoder.device}")
    print(f"Embedding dim: {encoder.get_embedding_dim()}")

    return encoder


def extract_embeddings(encoder, images, captions):
    """Extract embeddings from images and captions using CLIP."""
    print("\nExtracting image embeddings...")
    image_embeddings = encoder.encode_images(images)
    print(f"Image embeddings shape: {image_embeddings.shape}")

    print("\nExtracting text embeddings...")
    text_embeddings = encoder.encode_texts(captions)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    return image_embeddings, text_embeddings


def compute_and_save_results(image_embeddings, text_embeddings, metadata, output_path):
    """Compute similarity and save all results."""
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_cosine_similarity(
        image_embeddings, text_embeddings)

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(
        f"Similarity range: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")

    diagonal_similarities = torch.diagonal(similarity_matrix)
    mean_diagonal = diagonal_similarities.mean().item()
    print(f"Mean diagonal similarity (correct pairs): {mean_diagonal:.4f}")

    print("\nSaving results...")
    save_embeddings(image_embeddings, metadata, str(output_path), "image")
    save_embeddings(text_embeddings, metadata, str(output_path), "text")
    save_similarity_matrix(similarity_matrix, metadata,
                           str(output_path), "similarity_matrix.csv")

    print(f"Results saved to: {output_path}")


def main():
    """Run CLIP analysis."""
    print("=" * 60)
    print("CLIP Analysis - Unified Multimodal Embeddings")
    print("=" * 60)

    dataset_path, output_path = setup_paths()

    print("\nLoading dataset...")
    images, captions, metadata = load_dataset(str(dataset_path))
    print(f"Loaded {len(images)} images and {len(captions)} captions")

    encoder = initialize_encoder()

    image_embeddings, text_embeddings = extract_embeddings(
        encoder, images, captions)

    compute_and_save_results(
        image_embeddings, text_embeddings, metadata, output_path)

    print("=" * 60)
    print("CLIP analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
