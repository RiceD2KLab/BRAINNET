'''Functions common to classification and segmentation'''

import numpy as np

SEGMENTS = {
    0: "ELSE",
    1: "NCR",
    2: "ED",
    3: "ET"
}

SEGMENT_COLORS = {
    0: "gray",
    1: "red",
    2: "green",
    3: "yellow"
}


BRATS_REGIONS = {
    "ELSE": [0],
    "WT": [1, 2, 3], # ET + NT + ED
    "TC": [1, 3], # ET + necrotic tumor
    "ET": [3]
}

def get_mri_subj_id(file_name):
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # subj_id: 00006
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0].split('-')[-1]
    return None

def get_mri_subj(file_name):
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # result: UPENN-GBM-00006
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0]
    return None
    
def get_mri_slice_file_name(file_name):
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # result: UPENN-GBM-00006_1.nii.gz
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0] + \
                    "_" + file_name.strip().split('_')[3]
    return None

def get_mri_file_no(file_name):
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # file_no: 11
    if file_name.strip().split('.')[-1] == 'gz':
        return int(file_name.strip().split('.')[0].split('_')[-1])
    return None

def get_largest_tumor_slice_idx(img_data, sum=False):
    non_zero_x = np.count_nonzero(img_data, axis=0)
    if sum is True:
        non_zero_x = np.sum(img_data, axis=0)
    total_y = np.sum(non_zero_x, axis=0 )
    slice_idx = np.argmax(total_y)
    return slice_idx, total_y[slice_idx]