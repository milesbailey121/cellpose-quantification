import numpy as np
from skimage import measure
import cv2
import pandas as pd
import tifffile
import os
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool, freeze_support, RLock

def minmax_normalise(data):
    """
    Normalize the data using min-max normalization.

    Parameters:
    data (list or array-like): List or array of values to be normalized.

    Returns:
    list: Min-max normalized data.
    """
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def log_normalise(data):
    """
    Normalize the data using logarithmic normalization.

    Parameters:
    data (list or array-like): List or array of values to be normalized.

    Returns:
    array: Logarithmically normalized data.
    """
    normalised_data = np.log10((data + 1))
    return normalised_data

def quantify_cell_features(args, pos):
    """
    Quantify cell features & intensities from segmented and raw images.

    Parameters:
    args (tuple): A tuple containing segmentation image, TIFF image, filename, normalization technique, and channel names.
    pos (int): Position indicator (used for progress bar).

    Returns:
    DataFrame: A DataFrame containing quantified cell features and mean intensity values.
    """
    seg_img, tiff_img, fname, norm, channel_names = args
    results = []


    unique_values = np.unique(seg_img)

    for label in tqdm(unique_values, desc=f"Quantifying {fname}"):
        if label == 0:
            continue

        cell_mask = np.uint16(seg_img == label)
        intensity_dict = {'Cell_ID': label}

        for channel in range(tiff_img.shape[0]):
            channel_img = tiff_img[channel, :, :]
            mean_intensity = np.mean(channel_img[cell_mask > 0])
            intensity_dict[channel_names[channel]] = mean_intensity

        cell_features = GetProps(cell_mask, tiff_img[0])
        cell_data = {**intensity_dict, **cell_features}
        cell_data['Filename'] = fname
        results.append(pd.DataFrame(cell_data, index=[0]))

    final_results = pd.concat(results, ignore_index=True)

    if norm:
        for column in final_results.columns:
            if column in channel_names:
                if norm == "minmax":
                    final_results[column] = minmax_normalise(final_results[column])
                elif norm == "log":
                    final_results[column] = log_normalise(final_results[column])

    return final_results

def GetProps(mask, image):
    """
    Extract properties of labeled regions in an image.

    Parameters:
    mask (ndarray): Binary mask of labeled regions.
    image (ndarray): Image from which properties are to be extracted.

    Returns:
    dict: Dictionary of properties for each labeled region.
    """
    properties = ['area', 'centroid', 'perimeter', 'eccentricity', 'solidity', 'orientation']
    # I have no idea why i have to rotate the masks and images before running it through measure.regionprops_table
    # Without rotating masks, the centroid positions will be rotated 90 degrees away from the original images????
    mask = np.rot90(mask,k=1,axes=(1,0))
    image = np.rot90(image,k=1,axes=(1,0))
    dat = measure.regionprops_table(mask, image, properties=properties)
    return dat

def ProcessDirectory(img_dir, mask_dir):
    """
    Process the directories to list all image and mask files.

    Parameters:
    img_dir (str): Path to the image directory.
    mask_dir (str): Path to the mask directory.

    Returns:
    tuple: Dictionaries mapping filenames to their paths for images and masks.
    """
    img_file_list = []
    mask_file_list = []

    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('.tif', '.tiff')):
                img_file_list.append(os.path.join(root, file))

    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(('.tif', '.tiff', '.png', '.npy')):
                mask_file_list.append(os.path.join(root, file))

    img_dict = {os.path.basename(key): key for key in img_file_list}
    mask_dict = {os.path.basename(key): key for key in mask_file_list}

    if len(img_dict) != len(mask_dict):
        print(f"ERROR: The number of images({len(img_dict)}) and masks({len(mask_dict)}) do not match")
        exit()

    return img_dict, mask_dict

def find_files(root_dir, extension):
    """
    Find all files with a given extension in a directory.

    Parameters:
    root_dir (str): Root directory to search.
    extension (str): File extension to look for.

    Returns:
    list: List of file paths matching the extension.
    """
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def WritetoFile(results, norm):
    """
    Write the results to a CSV file.

    Parameters:
    results (DataFrame): DataFrame containing the results.
    norm (str or bool): Normalization technique or False if not normalized.

    Returns:
    None
    """
    fpath = "cell_intensity_results_normalised.csv" if isinstance(norm, str) else "cell_intensity_results.csv"
    results.to_csv(fpath, index=False)

def main(image_directory, mask_directory, marker_path, normalization):
    """
    Main function to process directories, quantify cell features, and write results to a file.

    Parameters:
    image_directory (str): Path to the image directory.
    mask_directory (str): Path to the mask directory.
    marker_path (str): Path to the marker CSV file.
    normalization (str or bool): Normalization technique or False if not normalized.

    Returns:
    None
    """
    img_dict, mask_dict = ProcessDirectory(image_directory, mask_directory)
    channel_names_df = pd.read_csv(marker_path)
    #Converts csv list column names into a list
    channel_names = channel_names_df.columns.values.tolist()
    results = []

    for img_file, mask_file in zip(img_dict.values(), mask_dict.values()):
        try:
            tiff_img = tifffile.imread(img_file)

            if mask_file.endswith(('.tif', '.tiff')):
                seg_mask = tifffile.imread(mask_file)
            elif mask_file.endswith('.npy'):
                #allow pickle is neccesary to access cellpose npy masks
                seg_mask = np.load(mask_file, allow_pickle=True).item()
                seg_mask = seg_mask["masks"]
            elif mask_file.endswith('.png'):
                seg_mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            else:
                print(f"{mask_file} is not an accepted file format (npy/tiff/png)")
                continue

            single_cell_image_list = [(seg_mask, tiff_img, os.path.basename(img_file), normalization, channel_names)]

            for args in single_cell_image_list:
                results.append(quantify_cell_features(args, pos=0))  # Position 0 indicates no progress bar

        except PermissionError as e:
            print(f"PermissionError: {e} for file {img_file} or {mask_file}")
        except Exception as e:
            print(f"Error: {e} for file {img_file} or {mask_file}")

    final_results = pd.concat(results, ignore_index=True)
    WritetoFile(final_results, normalization)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print("Usage: python RunQuantification-CLI.py <image_directory> <mask_directory> <marker_path> <normalization>")
    else:
        image_directory = sys.argv[1]
        mask_directory = sys.argv[2]
        marker_path = sys.argv[3]
        normalization = sys.argv[4]
        main(image_directory, mask_directory, marker_path, normalization)
