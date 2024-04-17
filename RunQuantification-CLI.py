import argparse
import os
from Quantification import main

def parse_args():
    parser = argparse.ArgumentParser(description="Quantify cell features from images and masks.")
    parser.add_argument("image_directory", help="Path to the directory containing images.")
    parser.add_argument("mask_directory", help="Path to the directory containing masks.")
    parser.add_argument("marker_path", help="Path to the markers CSV file.")
    #parser.add_argument("--normalisation", default=False, action="store_true", help="Flag to perform normalization.")
    parser.add_argument("normalisation", help="Normalisation Technique")

    return parser.parse_args()

def validate_args(image_directory, mask_directory, marker_path,normalisation):
    image_directory = os.path.abspath(image_directory)
    mask_directory = os.path.abspath(mask_directory)
    marker_path = os.path.abspath(marker_path)

    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory '{image_directory}' does not exist.")
    if not os.path.exists(mask_directory):
        raise FileNotFoundError(f"Mask directory '{mask_directory}' does not exist.")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(f"Marker CSV file '{marker_path}' does not exist.")
    
    match normalisation:
        case "minmax":
            pass
        case "z-score":
            pass
        case "boxcox":
            pass
        case _:
            raise ValueError(f"{normalisation} is not supported or not formatted corretly, please use minmax or z-score normalisation")
            
        

    return image_directory, mask_directory, marker_path, normalisation

if __name__ == '__main__':
    args = parse_args()
    image_directory, mask_directory, marker_path, normalisation = validate_args(args.image_directory, args.mask_directory, args.marker_path,args.normalisation)
    main(image_directory, mask_directory, marker_path, normalisation)

