'''Functions common to classification and segmentation'''

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import nibabel as nib
import shutil

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

    # loop and read scans to add them to collated_images
    for idx1, subj_file in enumerate(subject_list):
        print(f"Working on subject no: {idx1 + 1} / {n_subjects}")
        for idx2, struct_scan in enumerate(struct_scan_list):
            # read in the scan
            scan_path = os.path.join(data_path, f"{subj_file}_{struct_scan}.nii.gz")
            scan = nib.load(scan_path).get_fdata()
            # insert the scan image array data
            collated_images[idx1, idx2, :, :, :] = scan[:, :, :]

    return collated_images


def get_data_stats(collated_images, subject_list, struct_scan_list, scan_type='struct', data_path=None):
    """
    Given a numpy array of collated scans generated using collated_scans, calculate statistics
    for normalizing the volumes. Measured statistics are mean, standard deviation, min, and max

    Inputs:
        collated_images - a numpy ndarray of shape (n_subjects, n_scans, height, width, depth)
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


def normalize_and_save(subjects_list, struct_scan_list, data_dir, output_dir, train_list, val_list, test_list):
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


def copy_segm_files(latent_space_norm_dir, segm_dir, subjects_list, train_list, val_list, test_list):
    """
    Copies associated SEGM annotation masks from reduced data dir to
    normalized latent space dir, sorted into train/val/test subdirs

    Inputs:
        latent_space_norm_dir - str path to normalized latent space data
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
            latent_space_norm_dir,
            train_list,
            val_list,
            test_list
        )
        dest_file = os.path.join(subdir, segm_name)

        # copy the file
        with open(source_file, 'rb') as input_file:
            with open(dest_file, 'wb') as output_file:
                shutil.copyfileobj(input_file, output_file)
