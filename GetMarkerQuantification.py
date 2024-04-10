import numpy as np
from skimage import measure
import cv2
import numpy as np
import pandas as pd
import tifffile
import os
from tqdm import tqdm

def z_score_normalize(data):
    mean_val = data.mean()
    std_dev = data.std()
    normalized_data = (data - mean_val) / std_dev
    return normalized_data

def quantify_cell_features(seg_img, tiff_img, fname):
    # Initialize DataFrame to store results
    results = []

    # Find unique pixel values in the image
    unique_values = np.unique(seg_img)

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(unique_values - 1), desc=f"Quantifing {fname}")

    # Iterate through each unique pixel value (excluding background)
    for label in unique_values:
        if label == 0:  # Skip background label
            continue
        
        # Mask out the current cell feature
        cell_mask = np.uint16(seg_img == label)

        # Initialize a dictionary to store intensities for each channel
        intensity_dict = {'Cell_ID': label}

        # Iterate through each channel
        for channel in range(tiff_img.shape[2]):
            # Extract channel from TIFF image
            channel_img = tiff_img[:, :, channel]

            # Calculate mean intensity within the cell feature for the current channel
            mean_intensity = np.mean(channel_img[cell_mask > 0])

            # Store mean intensity for the current channel
            intensity_dict[channel_names[channel]] = mean_intensity
    
        # Calculate cell features using MaskChannel function
        cell_features = GetProps(cell_mask, tiff_img)

        # Merge intensity and cell features data
        cell_data = {**intensity_dict, **cell_features}
        
        # Add a new column for the filename
        cell_data['Filename'] = fname

        # Append results for the current cell to the list
        results.append(pd.DataFrame(cell_data, index=[0]))

        # Update progress bar
        pbar.update(1)

    pbar.close()  # Close progress bar

    # Concatenate all DataFrames in the list to create the final DataFrame
    final_results = pd.concat(results, ignore_index=True)

    for column in final_results.columns:
        if column in channel_names:
            final_results[column] = z_score_normalize(final_results[column])

    return final_results


def GetProps(mask, image):
    """Function for quantifying a single channel image

    Returns a table with properites of each cell"""
    properties = ['area', 'centroid', 'perimeter', 'eccentricity', 'solidity', 'orientation']
    dat = measure.regionprops_table(
        mask, image,
        properties=properties
    )
    return dat

def ProcessDirectory(img_dir,mask_dir):
    img_file_list = os.listdir(img_dir)
    mask_file_list = os.listdir(mask_dir)
    img_dict = {key: None for key in img_file_list}
    mask_dict = {key: None for key in mask_file_list}
    return img_dict, mask_dict

def FileReader(img_dir, mask_dir, img_dict, mask_dict, channel_name_fpath):
    global channel_names
    channel_names_df = pd.read_csv(channel_name_fpath)
    channel_names = channel_names_df.columns.values.tolist()

    for key in img_dict.keys():
        # Read TIFF file
        tiff_img = tifffile.imread(os.path.join(img_dir,key))
        tiff_img = tiff_img.transpose(1,2,0)
        img_dict[key] = tiff_img

    for key in mask_dict.keys():
        # Read Mask file
        if key.__contains__(".tif") or key.__contains__(".tiff"):
            seg_mask = cv2.imread(os.path.join(mask_dir,key), cv2.IMREAD_GRAYSCALE)
        elif key.__contains__(".npy"):
            seg_mask = np.load(os.path.join(mask_dir,key), allow_pickle=True).item()
            seg_mask = seg_mask["masks"]
        else:
            print(f"{key} is not a accepted file format(npy or tiff)")
        mask_dict[key] = seg_mask


    return img_dict, mask_dict
    

def WritetoFile(results):
    # excel_fpath = "cell_intensity_results_normalised.xlsx"
    excel_fpath = "cell_intensity_results.xlsx"
    results.to_excel(excel_fpath, index=False)

# Example usage

def main():
    global img_dict,mask_dict 
    img_dir = "data/image"
    mask_dir = "data/mask"
    channel_name_fpath = r"markers.csv"
    img_dict,mask_dict = ProcessDirectory(img_dir,mask_dir)
    img_dict,mask_dict = FileReader(img_dir, mask_dir, img_dict, mask_dict, channel_name_fpath)
    result_list = []
    
    for key1 in img_dict.keys():
        for key2 in mask_dict.keys():
            # Split key2 to remove "_seg" from the filename
            base_filename = os.path.splitext(key2)[0]
            if os.path.splitext(key1)[0] == base_filename.split("_seg")[0]:
                result_list.append(quantify_cell_features(mask_dict[key2], img_dict[key1], key1))
            else:
                pass

    final_results = pd.concat(result_list, ignore_index=True)
    # Write results to Excel file
    WritetoFile(final_results)

if __name__ == '__main__':
    main()