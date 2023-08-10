'''Functions common to classification and segmentation'''

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import nibabel as nib
import shutil
import time

from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from tqdm.auto import tqdm
from typing import List, Tuple

# Map mask annotation labels to medical terms
SEGMENTS = {
    0: "ELSE",
    1: "NCR",
    2: "ED",
    3: "ET"
}

# Map mask annotation labels to colors for visualization
SEGMENT_COLORS = {
    0: "gray",
    1: "red",
    2: "green",
    3: "yellow"
}

# Map total slices for each slice direction
SLICE_TOTAL = {
    "DEPTH": 146,
    "CROSS_SIDE": 163,
    "CROSS_FRONT": 193
}

# Enum class for slice direction
class SliceDirection(str, Enum):
    DEPTH = "DEPTH"
    CROSS_SIDE = "CROSS_SIDE"
    CROSS_FRONT = "CROSS_FRONT"

def get_mri_subj_id(file_name):
    """
    Given a file name of an MRI 3D volume, extract
    the unique patient id number

    Ex: file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    subj_id: 00006

    Inputs:
        file_name - str, represents a NIFTI1 format volume

    Returns a string or None.
    """
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # subj_id: 00006
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0].split('-')[-1]
    return None


def get_mri_subj(file_name):
    """
    Given a file name of an MRI 3D volume, extract
    the entire subject name

    Ex: file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    result: UPENN-GBM-00006

    Inputs:
        file_name - str, represents a NIFTI1 format volume

    Returns a string or None.
    """
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # result: UPENN-GBM-00006
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0]
    return None


def get_mri_slice_file_name(file_name):
    """
    Given a file name of an MRI 2D image slice, extract
    the subject name and the slice index number for
    determining how many unique 2D slices exist for
    a patient.

    Ex: UPENN-GMB-00006_11_FLAIR_1.nii.gz
    result: UPENN-GBM-00006_1.nii.gz

    Inputs:
        file_name - str, represents a NIFTI1 format volume

    Returns a string or None.
    """
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # result: UPENN-GBM-00006_1.nii.gz
    if file_name.strip().split('.')[-1] == 'gz':
        return file_name.strip().split('_')[0] + \
                    "_" + file_name.strip().split('_')[-1]
    return None


def get_mri_file_no(file_name):
    """
    Given a file name of an MRI 3D volume, extract the
    patient's image status code.

    Code 11 indicates pre-tumor resection (surgical removal)
    Code 21 indicates post-tumor resection

    Ex: file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    file_no: 11

    Inputs:
        file_name - str, represents a NIFTI format volume
    """
    # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
    # file_no: 11
    if file_name.strip().split('.')[-1] == 'gz':
        return int(file_name.strip().split('.')[0].split('_')[-1])
    return None


def get_largest_tumor_slice_idx(img_data, sum=False):
    """
    Determines the index of the slice with the largest tumor area.
    Default method is to count nonzero pixels. Optionally can use
    summation.

    Inputs:
        img_data - numpy ndarray
        sum - bool (default False),
    Returns a tuple of slice_idx (int) and number of non-zero pixels (int)
    """
    non_zero_x = np.count_nonzero(img_data, axis=0)
    if sum is True:
        non_zero_x = np.sum(img_data, axis=0)
    total_y = np.sum(non_zero_x, axis=0 )
    slice_idx = np.argmax(total_y)
    return slice_idx, total_y[slice_idx]

def create_train_dir_by_date():
    """
    Creates training directory dynamically
    using format maskformer/{unix_date}

    Returns the path created.
    """
    now = str(int( time.time() ))
    return os.path.join("maskformer", now)

def split_subjects(input_list, split_ratio=0.8, seed=0):
    """
    Given a list of strings representing unique patients,
    splits the list into train/val/test sets

    Inputs:
        input_list - a list of strings of unique patient ids
        split_ratio - float, default 0.8
        seed - int, default 0, used for setting random seed

    Returns a tuple of lists
    """
    random.seed(seed)

    # get the number of records for output files
    n_data = len(input_list)
    num_first_split = round(n_data * split_ratio)
    num_second_list = n_data - num_first_split

    # generate the output lists
    output_list_1 = random.sample(input_list, num_first_split)
    output_list_2 = [
        data for data in input_list if data not in output_list_1
    ]

    return output_list_1, output_list_2


def collate_scans(subject_list, shape, struct_scan_list, scan_type='struct', data_path=None):
    """
    Combines all structural scans across all patients into a single numpy ndarray

    Inputs:
        subject_list - a list of strings representing unique patients
        shape - a tuple representing the dimensions of a single structural scan
        struct_scan_list - a list of strings containing the different types of scans
        scan_type - str, default 'struct' or 'latent_vector'
        data_path - str, path to data dir, default None

    Returns a numpy ndarray of shape (n_subjects, n_scans, height, width, depth)
    """
    # confirm a data_path is given
    assert data_path is not None

    # set the number of subjects
    n_subjects = len(subject_list)

    # set the image dimensions
    n_height, n_width, n_depth = shape

    # determine how many scans to expect
    assert scan_type == 'struct' or scan_type == 'latent_vector'
    if scan_type == 'struct':
        n_scans = 4
        assert len(struct_scan_list) == 4
    else:
        n_scans = 3
        assert len(struct_scan_list) == 3

    # initialize an empty numpy array to hold all data
    collated_shape = (n_subjects, n_scans, n_height, n_width, n_depth)
    collated_images = np.zeros(collated_shape)

    # TODO: check pattern
    # loop and read scans to add them to collated_images
    for idx1, subj_file in enumerate(subject_list):
        print(f"Working on subject no: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            # read in the scan
            scan_path = os.path.join(data_path, f"{subj_file}_{struct_scan}.nii.gz")
            if scan_type == 'struct':
                scan_path = os.path.join(data_path, f"{subj_file}_11_{struct_scan}_cut.nii.gz")

            scan = nib.load(scan_path).get_fdata()
            # insert the scan image array data
            collated_images[idx1, idx2, :, :, :] = scan[:, :, :]

    return collated_images


def get_data_stats(subject_list, struct_scan_list, scan_type='struct', data_path=None):
    """
    Given a numpy array of collated scans generated using collated_scans, calculate statistics
    for normalizing the volumes. Measured statistics are mean, standard deviation, min, and max

    Inputs:
        subject_list - a list of strings of unique subjects
        struct_scan_list - a list of strings defining types of scans
        scan_type - a str, default 'struct' or 'latent_vector'
        data_path - a str, path to data dir

    Returns a numpy ndarray of shape (n_subjects, n_scans, 4)
    """
    # confirm a data_path is given
    assert data_path is not None

    # set the number of subjects
    n_subjects = len(subject_list)

    # determine how many scans to expect
    assert scan_type == 'struct' or scan_type == 'latent_vector'
    if scan_type == 'struct':
        n_scans = 4
        assert len(struct_scan_list) == 4
    else:
        n_scans = 3
        assert len(struct_scan_list) == 3

    # initialize an empty numpy array to hold stats
    stats_arr_shape = (n_subjects, n_scans, 4)
    stats_arr = np.zeros(stats_arr_shape)

    # loop and calculate stats
    for idx1, subj_file in enumerate(subject_list):
        print(f"Working on Subject No: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            # read in the scan
            scan_path = os.path.join(data_path, f"{subj_file}_{struct_scan}.nii.gz")
            if scan_type == 'struct':
                scan_path = os.path.join(data_path, f"{subj_file}_11_{struct_scan}_cut.nii.gz")

            scan = nib.load(scan_path).get_fdata()
            # measure stats
            scan_mean = scan.mean()
            scan_std = scan.std()
            scan_min = scan.min()
            scan_max = scan.max()
            # insert into stats_arr
            stats_arr[idx1, idx2, 0] = scan_mean
            stats_arr[idx1, idx2, 1] = scan_std
            stats_arr[idx1, idx2, 2] = scan_min
            stats_arr[idx1, idx2, 3] = scan_max

    return stats_arr


def normalize_collated_data(collated_images, stats_arr, subject_list, struct_scan_list):
    """
    Max normalizes the volumes to range between 0 and 1. Mutates collated_images

    Inputs:
        collated_images - numpy ndarray of shape (n_subjects, n_scans, height, width, depth)
        stats_arr - numpy ndarray output from get_data_stats of shape (n_subjects, n_scans, 4)
        subject_list - list of strings of unique subject ids
        struct_scan_list - list of strings of scan types

    Returns None.
    """
    # set the number of subjects
    n_subjects = len(subject_list)

    # iterate over collated_images and max normalize
    for idx1, subj_file in enumerate(subject_list):
        print(f"Working on Subject No: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            collated_images[idx1, idx2, :, :, :] /= stats_arr[idx1, idx2, 3]


def update_collated_stats(collated_images, subjects_list, struct_scan_list):
    """
    Updates statistics after normalizing collated_images array.

    Inputs:
        collated_images - numpy ndarray of shape (n_subjects, n_scans, height, width, depth)
        subject_list - list of strings of unique subject ids
        struct_scan_list - list of strings of scan types

    Returns a numpy ndarray of shape (n_subjects, n_scans, 4)
    """
    # set the number of subjects
    n_subjects = len(subjects_list)

    # set the number of scans
    n_scans = len(struct_scan_list)

    # initialize an empty numpy array to hold stats
    stats_arr_shape = (n_subjects, n_scans, 4)
    stats_arr = np.zeros(stats_arr_shape)

    # iterate over collated_images and update stats_arr
    for idx1, subj_file in enumerate(subjects_list):
        print(f"Working on Subject No: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            # get the status for the current scan
            scan_mean = collated_images[idx1, idx2, :, :, :].mean()
            scan_std = collated_images[idx1, idx2, :, :, :].std()
            scan_min = collated_images[idx1, idx2, :, :, :].min()
            scan_max = collated_images[idx1, idx2, :, :, :].max()
            # insert into stats_arr
            stats_arr[idx1, idx2, 0] = scan_mean
            stats_arr[idx1, idx2, 1] = scan_std
            stats_arr[idx1, idx2, 2] = scan_min
            stats_arr[idx1, idx2, 3] = scan_max

    return stats_arr


def plot_scan_stats(stats_arr, struct_scan_list, figsize=(20, 4)):
    """
    Plots scan statistics measured in utils.mri_common.get_data_stats

    Inputs:
        stats_arr - numpy ndarray returned from utils.mri_common.get_data_stats
        struct_scan_list - a list of strings of scan types
        figsize - tuple of ints for figure size, default is (20, 4)

    Returns None
    """
    # set number of scans
    n_scans = len(struct_scan_list)

    # build the figure
    fig, axes = plt.subplots(nrows=1, ncols=n_scans, figsize=figsize)

    for idx, axs in enumerate(axes):
        axs.plot(stats_arr[:, idx, 0], label='mean')
        axs.plot(stats_arr[:, idx, 1], label='std')
        axs.plot(stats_arr[:, idx, 2], label='min')
        axs.plot(stats_arr[:, idx, 3], label='max')
        axs.set_title(f"stats for {struct_scan_list[idx]}")
        axs.legend()
        axs.set_xlabel('Patient ID')
        axs.set_ylabel('Value')

    plt.show()


def get_subdir(subj, target_dir, train_list, val_list, test_list):
    """
    Helper function to identify which subdirectory
    a given subject should be placed according to
    the train/val/test split lists

    Inputs -
        subj - str, unique subject id
        target_dir - str, top-level directory containing train/val/test subdirs
        train_list - list of strings of unique subject ids in train set
        val_list - list of strings of unique subject ids in val set
        test_list - list of strings of unique subject ids in test set

    Returns a str
    """
    if subj in train_list:
        return os.path.join(target_dir, "train")
    elif subj in val_list:
        return os.path.join(target_dir, "val")
    else:
        return os.path.join(target_dir, "test")


def normalize_and_save(subjects_list, struct_scan_list, data_dir, output_dir, train_list, val_list, test_list, scan_type='struct'):
    """
    Normalizes all structural scans and saves them to disk,
    sorted into subdirectories train/val/test depending on
    which set the subject belongs.

    Inputs:
        subjects_list - list of strings of unique subject ids
        struct_scan_list - list of strings representing scan types
        data_dir - str path to data dir
        output_dir - str path to output directory
        train_list - list of strings of unique subject ids in train set
        val_list - list of strings of unique subject ids in val set
        test_list - list of strings of unique subject ids in test set
        scan_type - str, default 'struct' or 'latent_vector'
    Returns None
    """
    # check if output dir exists, and if not, make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # make subdirectories for train/val/test data
        os.makedirs(os.path.join(output_dir, "train"))
        os.makedirs(os.path.join(output_dir, "val"))
        os.makedirs(os.path.join(output_dir, "test"))

    # set the number of subjects
    n_subjects = len(subjects_list)

    # set the number of scan
    n_scans = len(struct_scan_list)

    # iterate over the subject list and structural scan list
    # load, normalize, and save scan
    for idx1, subj_file in enumerate(subjects_list):
        print(f"Working on subject no: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            # load scan
            scan_path = os.path.join(data_dir, f"{subj_file}_{struct_scan}.nii.gz")
            if scan_type == 'struct':
                scan_path = os.path.join(data_dir, f"{subj_file}_11_{struct_scan}_cut.nii.gz")

            scan = nib.load(scan_path)
            scan_fdata = scan.get_fdata()

            # capture affine transformation matrix
            affine = scan.affine

            # capture header
            header = scan.header

            # find the max value
            max_val = scan_fdata.max()

            # normalize cube by max
            scan_norm = scan_fdata / max_val

            # setup output file
            output_scan = nib.Nifti1Image(scan_norm, affine, header)

            # get train/val/test subdirectory for current subj
            subdir = get_subdir(
                subj_file,
                output_dir,
                train_list,
                val_list,
                test_list
            )
            output_scan_path = os.path.join(subdir, f"{subj_file}_11_{struct_scan}.nii.gz")
            # write to disk
            nib.save(output_scan, output_scan_path)


def copy_segm_files(norm_dir, segm_dir, subjects_list, train_list, val_list, test_list):
    """
    Copies associated SEGM annotation masks from reduced data dir to
    normalized latent space dir, sorted into train/val/test subdirs

    Inputs:
        norm_dir - str path to normalized data
        segm_dir - str path to reduced dims data containing segm masks
        subjects_list - list of strings of unique subject ids
        train_list - list of strings of unique subject ids in train set
        val_list - list of strings of unique subject ids in val set
        test_list - list of strings of unique subject ids in test set

    Returns None
    """
    # create empty list to track segm files
    segm_files = []

    # get a listing of the reduced files and sort
    reduced_files = os.listdir(segm_dir)
    reduced_files.sort()

    # iterate through the reduced files dir
    for nii in reduced_files:
        if nii.split('_')[2] == 'segm':
            segm_files.append(nii)

    # check that the number of segm masks is the same
    # as the number of subjects
    assert len(segm_files) == len(subjects_list)

    # copy segm files into latent space normalized dir
    # note that segm_files and subjects_list are sorted
    # so we are guaranteed that they will match 1-1
    for segm, subj in zip(segm_files, subjects_list):
        # specify where the file is coming from
        source_file = os.path.join(segm_dir, segm)

        # specify the new name
        # UPENN-GBM-XXXXX_11_segm.nii.gz
        segm_split_name = segm.split('_')
        segm_name = f"{segm_split_name[0]}_11_segm.nii.gz"

        # specify the output directory
        subdir = get_subdir(
            subj,
            norm_dir,
            train_list,
            val_list,
            test_list
        )
        dest_file = os.path.join(subdir, segm_name)

        # copy the file
        with open(source_file, 'rb') as input_file:
            with open(dest_file, 'wb') as output_file:
                shutil.copyfileobj(input_file, output_file)

def get_zero_reduction_dimensions(mri_dir: str, subj_list:List[str], struct_scan_list:List[str]):
    """
    Given input data, determines new dimensions by clipping out voxels that are entirely zero

    Args:
        mri_dir (str): path to data
        subj_list (list): list of subject ids, as strings
        struct_scan_list (list): names of input scan types, as strings

    Returns:
        tuple of ints representing indices for clipping cube boundaries
    """

    # note: this preprocessing assumes that the folder structure is in fixed in this format
    # mri_dir/UPENN-GBM-00001_11/UPENN-GBM-00001_11_FLAIR.nii.gz

    # load flair image and use its shape to store zero data information
    subj_file = subj_list[0]
    file_name = f"{subj_file}_11_FLAIR.nii.gz"

    file_path = os.path.join(mri_dir, f"{subj_file}_11", file_name)
    img_data = nib.load(file_path).get_fdata()
    n0 = img_data.shape[0]
    n1 = img_data.shape[1]
    n2 = img_data.shape[2]

    n_sample = len(subj_list)
    all_zero_a0 = np.zeros( (n0, n_sample) )
    all_zero_a1 = np.zeros( (n1, n_sample) )
    all_zero_a2 = np.zeros( (n2, n_sample) )

    a0_min_idx = n0
    a0_max_idx = 0
    a1_min_idx = n1
    a1_max_idx = 0
    a2_min_idx = n2
    a2_max_idx = 0

    # find non-zero slides
    for struct_scan in struct_scan_list:
        for idx in range(n_sample):
            # get the current image
            # extract the subject scan for the first manually-revised segmentation label
            subj_file = subj_list[idx]
            file_name = f"{subj_file}_11_{struct_scan}.nii.gz"
            file_path = os.path.join(mri_dir, f"{subj_file}_11", file_name)
            img_data = nib.load(file_path).get_fdata()

            # find all zero lines in each of the 3 dimensions
            all_zero_a01 = np.all(img_data == 0, axis=2)
            all_zero_a0[:,idx]  = np.all(all_zero_a01 == True, axis=1).astype(int)
            all_zero_a1[:,idx]  = np.all(all_zero_a01 == True, axis=0).astype(int)

            all_zero_a02 = np.all(img_data == 0, axis=1)
            all_zero_a2[:,idx]  = np.all(all_zero_a02 == True, axis=0).astype(int)

        # find all zero planes in each of the 3 dimensions
        a0_empty = np.all(all_zero_a0 == True, axis=1)
        a1_empty = np.all(all_zero_a1 == True, axis=1)
        a2_empty = np.all(all_zero_a2 == True, axis=1)

        # find new bound values
        a0_min_idx = np.min( (a0_min_idx, np.where(~a0_empty)[0].min()) )
        a0_max_idx = np.max( (a0_max_idx, np.where(~a0_empty)[0].max()) )
        a1_min_idx = np.min( (a1_min_idx, np.where(~a1_empty)[0].min()) )
        a1_max_idx = np.max( (a1_max_idx, np.where(~a1_empty)[0].max()) )
        a2_min_idx = np.min( (a2_min_idx, np.where(~a2_empty)[0].min()) )
        a2_max_idx = np.max( (a2_max_idx, np.where(~a2_empty)[0].max()) )

    print("min idx in height is:",a0_min_idx,"max idx in height is:",a0_max_idx)
    print("min idx in width is :",a1_min_idx,"max idx in width is :",a1_max_idx)
    print("min idx in depth is :",a2_min_idx,"max idx in depth is :",a2_max_idx)

    n0_new = a0_max_idx - a0_min_idx + 1
    n1_new = a1_max_idx - a1_min_idx + 1
    n2_new = a2_max_idx - a2_min_idx + 1
    image_size_ratio = (a0_max_idx-a0_min_idx+1) * (a1_max_idx-a1_min_idx+1) * (a2_max_idx-a2_min_idx+1) / (n0 * n1 * n2)

    print("Original height / width / depth :", n0, "/", n1, "/", n2)
    print("     New height / width / depth :", n0_new, "/", n1_new, "/", n2_new)
    print("Data reduction :", round((1-image_size_ratio)*100, 2), "%")

    return (
        a0_min_idx,
        a0_max_idx,
        a1_min_idx,
        a1_max_idx,
        a2_min_idx,
        a2_max_idx
    )

# create a new folder, save all reduced data into the new folder
def reduce_data(new_dim:  Tuple[int, int, int, int, int, int],
                subj_list: List[str], struct_scans: List[str],
                struct_scan_dir: str, segm_dir:str, output_dir:str):
    """
    Crops cubes to reduced dimensions to eliminate non-zero voxels

    Args:
        new_dim (tuple): new dimensions, output by get_zero_reduction_dimensions()
        subj_list (list): list of unique subject ids, as strings
        struct_scans (list): list of scan types, as strings
        struct_scan_dir (str): path to structural scan volumes
        segm_dir (str): path to segmentation volumes
        output_dir (str): path to save dimensionally-cropped data

    Returns:
        None.
    """

    (a0_min_idx, a0_max_idx,
     a1_min_idx, a1_max_idx,
     a2_min_idx, a2_max_idx) = new_dim

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sliced_nifti = None

    for idx in range(len(subj_list)):

        subj_file = subj_list[idx]

        # working on structure files
        for struct_scan in struct_scans:
            # obtain file name
            file_name = f"{subj_file}_11_{struct_scan}.nii.gz"
            file_path = os.path.join(struct_scan_dir, f"{subj_file}_11", file_name)
            nifti = nib.load(file_path)

            # reduce data
            sliced_data = nifti.get_fdata()[a0_min_idx:a0_max_idx, a1_min_idx:a1_max_idx, a2_min_idx:a2_max_idx]
            sliced_nifti = nib.Nifti1Image(sliced_data, nifti.affine, nifti.header)

            # save reduced data
            # UPENN-GBM-00002_11_FLAIR.nii.gz
            output_file = os.path.join(output_dir, f"{subj_file}_11_{struct_scan}_cut.nii.gz")
            nib.save(sliced_nifti, output_file)

        # working on segm files
        file_name = f"{subj_file}_11_segm.nii.gz"
        file_path = os.path.join(segm_dir, file_name)
        nifti = nib.load(file_path)

        # reduce data
        sliced_data = nifti.get_fdata()[a0_min_idx:a0_max_idx, a1_min_idx:a1_max_idx, a2_min_idx:a2_max_idx]
        sliced_nifti = nib.Nifti1Image(sliced_data, nifti.affine, nifti.header)

        # save reduced data
        output_file = os.path.join(output_dir, f"{subj_file}_11_segm_cut.nii.gz")
        nib.save(sliced_nifti, output_file)

    # print dimensions of last image
    n_h = sliced_data.shape[0]
    n_w = sliced_data.shape[1]
    n_d = sliced_data.shape[2]
    print('Dimension of Image are height:',n_h,'width:',n_w,'depth:',n_d)

def generate_2d_slices(input_dir: str = None, output_dir: str = None, orientation: SliceDirection = None):
    """
    Extracts 2d slices from a loaded 3D volumes in the specified orientation

    orientation can be either "DEPTH", "CROSS_FRONT", or "CROSS_SIDE"

    Returns None.
    """

    # specify directory paths
    direction = orientation.name.lower()
    output_train_dir = os.path.join(output_dir, "train", direction)
    output_val_dir = os.path.join(output_dir, "val", direction)
    output_test_dir = os.path.join(output_dir, "test", direction)

    # create the directories, assume if train_dir does not exist
    # then neither will val_dir nor test_dir
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
        os.makedirs(output_val_dir)
        os.makedirs(output_test_dir)

    assert os.path.exists(input_dir)

    # specify the train/val/test dirs for the loaded 3d mri
    # utils.mri_common.normalize_and_save specifies the subdirs
    # as train/val/test so as long as the volumes were generated
    # using the function, we can expect the following structure:
    # e.g. latent_space_vectors_annot_reduced_norm/train
    # images_annot_reduced_norm/train

    loaded_3d_mri_dir_train = os.path.join(input_dir, "train")
    loaded_3d_mri_dir_val = os.path.join(input_dir, "val")
    loaded_3d_mri_dir_test = os.path.join(input_dir, "test")
    print("loaded_3d_mri_dir_train", loaded_3d_mri_dir_train)
    print("loaded_3d_mri_dir_val", loaded_3d_mri_dir_val)
    print("loaded_3d_mri_dir_test", loaded_3d_mri_dir_test)

    # iterate over train/val/test dirs and extract 2d slices
    input_dir_list = [loaded_3d_mri_dir_train, loaded_3d_mri_dir_val, loaded_3d_mri_dir_test]
    output_dir_list = [output_train_dir, output_val_dir, output_test_dir]
    print('Extracting 2D slices from 3D volumes (now is a good time to take 5 min coffee break ...)')

    for in_subdir, out_subdir in tqdm(zip(input_dir_list, output_dir_list), total=len(input_dir_list)):
        print(f"Working on {in_subdir} --> {out_subdir}")

        _extract_2d_slices(
            orientation=orientation,
            input_dir=in_subdir,
            output_dir=out_subdir
        )

    return output_dir_list

def _extract_2d_slices(input_dir: str, output_dir: str, orientation: SliceDirection):
    """
    helper function to generate_2d_slices
    """
    # get a listing of files in the input directory
    dir_list = os.listdir(input_dir)

    # call _process_volume to create slices in parallel for each file
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(_process_volume,
                            dir_list,
                            [input_dir]*len(dir_list),
                            [output_dir]*len(dir_list),
                            [orientation]*len(dir_list)),
                    total=len(dir_list)))

def _process_volume(infile, input_dir, output_dir, orientation):
    """
    helper function called to generate 2d slices for 1 volume/1 patient
    """
    # load the volume
    nifti = nib.load(os.path.join(input_dir, infile))
    # get the affine transformation matrix
    affine = nifti.affine
    # get the header
    header = nifti.header
    # get the dimensions
    n_height, n_width, n_depth = tuple(nifti.header["dim"][1:4])

    # determine expected # of slices and execute process in parallel
    if orientation == SliceDirection.DEPTH:
        # idx_range = range(n_depth)
        for idx in range(n_depth):
            sliced_data = nifti.get_fdata()[:, :, idx]
            _process_slice(idx, sliced_data, affine, header, infile, output_dir)

    elif orientation == SliceDirection.CROSS_SIDE:
        for idx in range(n_height):
            sliced_data = nifti.get_fdata()[idx, :, :]
            _process_slice(idx, sliced_data, affine, header, infile, output_dir)
    elif orientation == SliceDirection.CROSS_FRONT:
        for idx in range(n_width):
            sliced_data = nifti.get_fdata()[:, idx, :]
            _process_slice(idx, sliced_data, affine, header, infile, output_dir)


def _process_slice(idx, sliced_data, affine, header, infile, output_dir):
    """
    helper function to save the a 2d slice
    """
    # convert numpy array to Nifti1Image format
    sliced_nifti = nib.Nifti1Image(sliced_data, affine, header)

    # save image
    save_fn = f"{infile.split('.nii.gz')[0]}_{idx}.nii.gz"
    nib.save(sliced_nifti, os.path.join(output_dir, save_fn))
