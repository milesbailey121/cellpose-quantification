from .cellpose_quantify import quantify
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Quantify cell features from images and masks.")
    parser.add_argument("image_directory", help="Path to the directory containing images.")
    parser.add_argument("mask_directory", help="Path to the directory containing masks.")
    parser.add_argument("marker_path", help="Path to the markers CSV file.")
    parser.add_argument("--normalisation", choices=["minmax", "log"], help="Normalisation technique to use. Choose 'minmax' or 'log'. If not provided, no normalisation is applied.", default=False)
    return parser.parse_args()

def validate_args(image_directory, mask_directory, marker_path, normalisation):
    image_directory = os.path.abspath(image_directory)
    mask_directory = os.path.abspath(mask_directory)
    marker_path = os.path.abspath(marker_path)

    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory '{image_directory}' does not exist.")
    if not os.path.exists(mask_directory):
        raise FileNotFoundError(f"Mask directory '{mask_directory}' does not exist.")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(f"Marker CSV file '{marker_path}' does not exist.")
    
    if normalisation and normalisation not in ["minmax", "log"]:
        raise ValueError(f"{normalisation} is not supported. Please use 'log' or 'minmax' normalisation.")
    
    return image_directory, mask_directory, marker_path, normalisation

def cli():
    """
    Entry point for CLI.
    """
    try:
        if len(sys.argv) != 5:
            print("Usage: python RunQuantification-CLI.py <image_directory> <mask_directory> <marker_path> <normalization>")
        else:
            args = parse_args()
            image_directory, mask_directory, marker_path, normalisation = validate_args(
                args.image_directory, args.mask_directory, args.marker_path, args.normalisation
            )
            quantify(image_directory, mask_directory, marker_path, normalisation)

    except Exception as e:
        print(f"Error: {e}",file=sys.stderr)

if __name__ == "__main__":
    cli()