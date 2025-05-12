import numpy as np
from skimage import measure
import cv2
import pandas as pd
import tifffile
import os
from tqdm import tqdm

def minmax_normalise(data):
    """
    normalise the data using min-max normalisation.

    Parameters:
    data (list or array-like): List or array of values to be normalised.

    Returns:
    list: Min-max normalised data.
    """
    min_val = min(data)
    max_val = max(data)
    normalised_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalised_data

def log_normalise(data):
    """
    normalise the data using logarithmic normalisation.

    Parameters:
    data (list or array-like): List or array of values to be normalised.

    Returns:
    array: Logarithmically normalised data.
    """
    # We add 1 to avoid log(0)
    normalised_data = np.log10((data + 1))
    return normalised_data

def get_cell_features(args):
    """
    Quantify cell features & intensities from segmented and raw images.

    Parameters:
    args (tuple): A tuple containing segmentation image, TIFF image, filename, normalisation technique, and channel names.
    pos (int): Position indicator (used for progress bar).

    Returns:
    DataFrame: A DataFrame containing quantified cell features and mean intensity values.
    """
    seg_img, tiff_img, fname, norm, channel_names = args
    results = []


    unique_values = np.unique(seg_img)

    for label in tqdm(unique_values, desc=f"Quantifying {fname}",ascii="░▒█",colour="GREEN"):
        if label == 0:
            continue

        cell_mask = np.uint16(seg_img == label)
        intensity_dict = {'Cell_ID': label}
        # Calculate mean intensity for each channel
        # Assuming tiff_img is a 3D array with shape (channels, height, width)
        for channel in range(tiff_img.shape[0]):
            channel_img = tiff_img[channel, :, :]
            mean_intensity = np.mean(channel_img[cell_mask > 0])
            intensity_dict[channel_names[channel]] = mean_intensity

        cell_features = get_props(cell_mask, tiff_img[0])
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

def get_props(mask, image):
    """
    Extract properties of labeled regions in an image.

    Parameters:
    mask (ndarray): Binary mask of labeled regions.
    image (ndarray): Image from which properties are to be extracted.

    Returns:
    dict: Dictionary of properties for each labeled cell.
    """
    properties = ['area', 'centroid', 'perimeter', 'eccentricity', 'solidity', 'orientation']
    # I have no idea why i have to rotate the masks and images before running it through measure.regionprops_table
    # Without rotating masks, the centroid positions will be rotated 90 degrees away from the original images????
    mask = np.rot90(mask,k=1,axes=(1,0))
    image = np.rot90(image,k=1,axes=(1,0))
    dat = measure.regionprops_table(mask, image, properties=properties)
    return dat

def process_directory(img_dir, mask_dir):
    """
    Process the directories to list all image and mask files.

    Parameters:
    img_dir (str): Path to the image directory, assumes images are in tiff format.
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

def write_to_file(results, norm):
    """
    Write the results to a CSV file.

    Parameters:
    results (DataFrame): DataFrame containing the results.
    norm (str or bool): Normalisation technique or False if not normalised.

    Returns:
    None
    """
    fpath = "cell_intensity_results_normalised.csv" if isinstance(norm, str) else "cell_intensity_results.csv"
    results.to_csv(fpath, index=False)

def run(image_directory, mask_directory, marker_path, normalisation):
    """
    Main function to process directories, quantify cell features, and write results to a file.

    Parameters:
    image_directory (str): Path to the image directory.
    mask_directory (str): Path to the mask directory.
    marker_path (str): Path to the marker CSV file.
    normalisation (str): normalisation technique or None if not normalised.

    """
    img_dict, mask_dict = process_directory(image_directory, mask_directory)
    # Reads marker csv file, assumes the fist row is the channel/marker names.
    channel_names = pd.read_csv(marker_path,header=None).values[0]
    results = []

    for img_file, mask_file in zip(img_dict.values(), mask_dict.values()):
        try:
            tiff_img = tifffile.imread(img_file)
            if tiff_img.shape[0] != len(channel_names):
                print(f"ERROR: The number of channels in {img_file} does not match the number of channels/markers in {marker_path}")
                continue

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
            
            if seg_mask.shape != tiff_img[0].shape:
                print(f"ERROR: The shape of the mask {mask_file} does not match the shape of the image {img_file}")
                continue
            
            arglist = [(seg_mask, tiff_img, os.path.basename(img_file), normalisation, channel_names)]

            for args in arglist:
                results.append(get_cell_features(args))  

        except PermissionError as e:
            print(f"PermissionError: {e} for file {img_file} or {mask_file}")
        except Exception as e:
            print(f"Error: {e} for file {img_file} or {mask_file}")

    final_results = pd.concat(results, ignore_index=True)
    write_to_file(final_results, normalisation)

