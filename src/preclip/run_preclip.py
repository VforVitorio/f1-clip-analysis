"""
Pre-CLIP analysis script.

Extracts embeddings using separate ResNet50 (images) and sentence-transformers (text) models,
then computes similarity and saves results.
"""

from preclip_utils import compute_cosine_similarity
from utils import load_dataset, save_embeddings, save_similarity_matrix
from preclip_models import ImageEncoder, TextEncoder
import sys
import torch
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_paths():
    """Set up dataset and output directories."""
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = project_root / "dataset"
    output_path = project_root / "results" / "preclip"

    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")

    return dataset_path, output_path


def initialize_encoders():
    """Initialize image and text encoders."""
    print("\nInitializing encoders...")
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()

    print(f"Image encoder device: {image_encoder.device}")
    print(f"Text encoder device: {text_encoder.device}")
    print(f"Image embedding dim: {image_encoder.get_embedding_dim()}")
    print(f"Text embedding dim: {text_encoder.get_embedding_dim()}")

    return image_encoder, text_encoder


def extract_embeddings(image_encoder, text_encoder, images, captions):
    """Extract embeddings from images and captions."""
    print("\nExtracting image embeddings...")
    image_embeddings = image_encoder.encode(images)
    print(f"Image embeddings shape: {image_embeddings.shape}")

    print("\nExtracting text embeddings...")
    text_embeddings = text_encoder.encode(captions)
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
    """Run Pre-CLIP analysis."""
    print("=" * 60)
    print("Pre-CLIP Analysis - Separate Image and Text Encoders")
    print("=" * 60)

    dataset_path, output_path = setup_paths()

    print("\nLoading dataset...")
    images, captions, metadata = load_dataset(str(dataset_path))
    print(f"Loaded {len(images)} images and {len(captions)} captions")

    image_encoder, text_encoder = initialize_encoders()

    image_embeddings, text_embeddings = extract_embeddings(
        image_encoder, text_encoder, images, captions
    )

    compute_and_save_results(
        image_embeddings, text_embeddings, metadata, output_path)

    print("=" * 60)
    print("Pre-CLIP analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
