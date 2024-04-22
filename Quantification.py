import numpy as np
from skimage import measure
import cv2
import numpy as np
import pandas as pd
import tifffile
import os
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool, freeze_support, RLock

def z_score_normalise(data):
    mean_val = data.mean()
    std_dev = data.std()
    normalized_data = (data - mean_val) / std_dev
    return normalized_data

def minmax_normalise(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def boxcox_normalise(data):
    data = data + 1
    normalised_data, fitted_lambda = stats.boxcox(data)
    return normalised_data

def quantify_cell_features(args,pos):
    seg_img, tiff_img, fname, norm, channel_names = args
    """Function for quantifying intensity values and merging with properties"""
    # Initialize DataFrame to store results
    results = []

    # Find unique pixel values in the image
    unique_values = np.unique(seg_img)

    # Initialize tqdm progress bar
    #pbar = tqdm(total=len(unique_values + 1), desc=f"Quantifing {fname}")


    with tqdm(total=len(unique_values), desc=f"Quantifing {fname}", position=pos+1) as pbar:
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



    # Concatenate all DataFrames in the list to create the final DataFrame
    final_results = pd.concat(results, ignore_index=True)

    match norm:
        case "minmax":
            for column in final_results.columns:
                if column in channel_names:
                    #Only normalize marker columns, not cell properties 
                    final_results[column] = minmax_normalise(final_results[column])
        case "z-score":
            for column in final_results.columns:
                if column in channel_names:
                    #Only normalize marker columns, not cell properties 
                    final_results[column] = z_score_normalise(final_results[column])

        case "boxcox":
            for column in final_results.columns:
                if column in channel_names:
                    #Only normalize marker columns, not cell properties 
                    final_results[column] = boxcox_normalise(final_results[column])

    return final_results


def GetProps(mask, image):
    """Function for quantifying a single channel image
    Returns a dataframe with general properites of each cell"""
    properties = ['area', 'centroid', 'perimeter', 'eccentricity', 'solidity', 'orientation']
    dat = measure.regionprops_table(
        mask, image,
        properties=properties
    )
    return dat

def ProcessDirectory(img_dir,mask_dir):
    """Lists files in directorys and assigns filename for each image and mask dictionary"""
    img_file_list = os.listdir(img_dir)
    mask_file_list = os.listdir(mask_dir)
    if len(img_file_list) == len(mask_file_list):
        img_dict = {key: None for key in img_file_list}
        mask_dict = {key: None for key in mask_file_list}
        return img_dict, mask_dict
    else:
        raise Exception("Mismatch number of files in image and mask directory")

def FileReader(img_dir, mask_dir, img_dict, mask_dict, channel_name_fpath):
    """Generates Image, mask dictionary and channel names"""
    channel_names_df = pd.read_csv(channel_name_fpath)
    # Converts marker names into a list 
    channel_names = channel_names_df.columns.values.tolist() 

    for key in img_dict.keys():
        # Read TIFF file
        tiff_img = tifffile.imread(os.path.join(img_dir,key))
        #Transpose (N,X,Y)
        tiff_img = tiff_img.transpose(1,2,0) # Need to update dynamically 
        img_dict[key] = tiff_img

    for key in mask_dict.keys():
        # Read Mask file
        if key.__contains__(".tif") or key.__contains__(".tiff") or key.__contains__(".png"):
            seg_mask = cv2.imread(os.path.join(mask_dir,key), cv2.IMREAD_GRAYSCALE)
        elif key.__contains__(".npy"):
            #Load _seg.npy files, requires items() due to pickled obj 
            seg_mask = np.load(os.path.join(mask_dir,key), allow_pickle=True).item()
            seg_mask = seg_mask["masks"]
        else:
            print(f"{key} is not a accepted file format(npy/tiff/png)")
        mask_dict[key] = seg_mask


    return img_dict, mask_dict, channel_names
    

def WritetoFile(results,norm):
    if not norm:#Empty string is seen as False
        fpath = "cell_intensity_results.csv"
    else:
        fpath = "cell_intensity_results_normalised.csv"
    results.to_csv(fpath, index=False)

# Example usage

def main(image_directory, mask_directory, marker_path, normalization):
    freeze_support()
    #Populates img_dict and mask_dict with filename as keys
    img_dict,mask_dict = ProcessDirectory(image_directory,mask_directory) 
    # Adds image and masks corresponding to file names, and also saves channel_name.csv 
    img_dict,mask_dict,channel_names = FileReader(image_directory, mask_directory, img_dict, mask_dict, marker_path)
    single_cell_image_list = []
    
    for pos,key1 in enumerate(img_dict.keys()):
        for key2 in mask_dict.keys():
            if "_seg" in key2:
                base_filename = os.path.splitext(key2)[0]
                if os.path.splitext(key1)[0] == base_filename.split("_seg")[0]:
                    single_cell_image_list.append((mask_dict[key2], img_dict[key1], key1, normalization,channel_names))
            else:
                if os.path.splitext(key1)[0] == os.path.splitext(key2)[0]:
                    single_cell_image_list.append((mask_dict[key2], img_dict[key1], key1, normalization, channel_names))


    # pool = Pool()
    # results = pool.map(quantify_cell_features, single_cell_image_list)
    # pool.close()
    # pool.join()

    pool = Pool(processes=os.cpu_count(),initargs=(RLock(),),initializer=tqdm.set_lock)
    jobs = [pool.apply_async(quantify_cell_features, args=(args,pos,)) for pos, args in enumerate(single_cell_image_list)]
    pool.close()

    results = [job.get() for job in jobs]
    print("\n" * (len(single_cell_image_list) + 1))
    final_results = pd.concat(results, ignore_index=True)
    # Write results to Excel file
    WritetoFile(final_results,normalization)

if __name__ == '__main__':
    main()
