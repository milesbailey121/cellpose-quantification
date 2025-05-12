import argparse
import os
import sys
from . import quantification

def validate_args(parser,image_directory, mask_directory, marker_path, normalisation):
    image_directory = os.path.abspath(image_directory)
    mask_directory = os.path.abspath(mask_directory)
    marker_path = os.path.abspath(marker_path)

    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory '{image_directory}' does not exist.")
    if not os.path.exists(mask_directory):
        raise FileNotFoundError(f"Mask directory '{mask_directory}' does not exist.")
    if not os.path.exists(marker_path):
        parser.error(f"Marker CSV file '{marker_path}' does not exist.")
        raise FileNotFoundError(f"Marker CSV file '{marker_path}' does not exist.")
    
    if normalisation and normalisation not in ["minmax", "log"]:
        raise ValueError(f"{normalisation} is not supported. Please use 'log' or 'minmax' normalisation(-n log/minmax or --norm log/minmax).")
    
    return image_directory, mask_directory, marker_path, normalisation

def cli(argv=None):
    """
    Entry point for CLI.
    """
    parser = argparse.ArgumentParser(description="Quantify cell features from images and masks.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0")
    parser.add_argument("image_directory", help="Path to the directory containing images.")
    parser.add_argument("mask_directory", help="Path to the directory containing masks.")
    parser.add_argument("marker_path", help="Path to the markers CSV file.")
    parser.add_argument("-n", "--norm",
                        nargs='?', const="None", 
                        choices=["log","minmax","None"],
                        help="Normalisation technique to use. Choose 'minmax' or 'log'. If not provided, no normalisation is applied.")


    args = parser.parse_args(argv)

    # Check if the arguments are provided
    image_directory, mask_directory, marker_path, normalisation = validate_args(
                parser,
                args.image_directory, args.mask_directory, args.marker_path, args.norm
        )
    
    quantification.run(image_directory, mask_directory, marker_path, normalisation)



if __name__ == "__main__":
    cli()