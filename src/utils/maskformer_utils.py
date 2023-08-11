'''Common functions used by maskformer (segmentation)'''
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# local imports
import utils.mri_common as mri_common

from utils.mri_common import SliceDirection
from utils.mri_plotter import MRIPlotter
mri_plt = MRIPlotter()

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


def scale_mask(mask):
    """
    binarizes an input mask and then scales values to range between
    0 and 255

    Args:
        mask (torch.Tensor or numpy.ndarray): the input mask to scale

    Returns:
        scaled torch.Tensor or numpy.ndarray
    """
    if isinstance(mask, torch.Tensor):
        visual_mask = (mask.bool().numpy() * 255).astype(np.uint8)
    else:
        visual_mask = (mask * 255).astype(np.uint8)
    return visual_mask
    # return Image.fromarray(visual_mask)

def descale_mask(mask):
    """
    Normalizes mask

    Args:
        mask (torch.Tensor or numpy.ndarray): input mask

    Returns:
        descaled torch.Tensor or numpy.ndarray
    """
    return (mask/255).astype(np.uint8)

def softmax(logits):
    """
    Defines 3D softmax activation function

    Args:
        logits (torch.Tensor): final layer output to prediction layer

    Returns:
        torch.Tensor
    """
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)

def softmax_2d(logits):
    """
    Defines 2D softmax activation function

    Args:
        logits (torch.Tensor): final layer output to prediction layer

    Returns:
        torch.tensor
    """
    e_x = np.exp(logits - np.max(logits, axis=(0, 1), keepdims=True))
    return e_x / np.sum(e_x, axis=(0, 1), keepdims=True)

def denormalize_img(img_data):
    """
    Restores image amplitude values to original scale

    Args:
        img_data (torch.Tensor or numpy.ndarray): input image normalized using ADE mean/stdev

    Returns:
        torch.Tensor or numpy.ndarray
    """
    denormalized_image = ( img_data * np.array(ADE_STD)[:, None, None] ) + np.array(ADE_MEAN)[:, None, None]
    denormalized_image = ( denormalized_image * 255 ).astype(np.uint8)
    denormalized_image = np.moveaxis(denormalized_image, 0, -1)
    return denormalized_image

def mask_to_segmentation(mask_labels, class_labels):
    """
    Flattens mask labels with shape (n_label, width, height) to a
    segmentation image with shape (width, height)

    Inputs:
        mask_labels (torch.Tensor or numpy.ndarray): mask labels
        class_labels (numpy.ndarray): class labels as ints

    Returns:
        torch.Tensor or numpy.ndarray
    """
    # map (num_labels, width, height) to segmentation (width, height)
    input_mask_mapping = {mask_idx: label_id for mask_idx, label_id in enumerate(class_labels)}
    input_mask_to_segm = np.vectorize(input_mask_mapping.get)(mask_labels.argmax(axis=0))
    return input_mask_to_segm.astype(np.uint8)

def post_proc_result_to_segmentation(results):
    """
    Map results['segmentation] to correct labels

    Args:
        results (torch.Tensor): predicted segmentation masks

    Returns:
        torch.Tensor
    """
    # map results['segmentation'] to correct labels
    # note that id represents obj instances and can be different from label id
    # [{'id': 0, 'label_id': 0, 'was_fused': False, 'score': 0.99242},
    # {'id': 1, 'label_id': 2, 'was_fused': False, 'score': 0.889586},
    # {'id': 2, 'label_id': 3, 'was_fused': False, 'score': 0.900965},
    # {'id': 3, 'label_id': 1, 'was_fused': False, 'score': 0.944047}]
    # segmentation result may also have -1. add this to the mapping

    pred_mask_mapping = {info['id']: info['label_id'] for info in results['segments_info']}
    pred_mask_mapping[-1] = -1
    pred_mask_to_segm = np.vectorize(pred_mask_mapping.get)(results['segmentation'])
    return pred_mask_to_segm

def get_label_dictionary(segments):
    """
    Generate mappings between segment IDs and labels for MaskFormer.

    Args:
        segments (dict): Dictionary of segment IDs and corresponding labels.

    Returns:
        tuple: Two dictionaries:
               - id2label (dict): Segment ID to label mapping.
               - label2id (dict): Label to segment ID mapping.
    """
    id2label = segments.copy()

    label2id = {}
    for key, value in id2label.items():
        label2id[value] = key
        
    return id2label, label2id

def get_file_idx_bounds(orientation:SliceDirection, num_slice=None):
    n_total = mri_common.SLICE_TOTAL[orientation.name]
    if num_slice is None:
        return 0, n_total + 1
    else:
        lower_bound = num_slice // 2
        upper_bound = num_slice - lower_bound
        
        file_no_min = (n_total//2) - lower_bound
        file_no_max = (n_total//2) + upper_bound
        return file_no_min, file_no_max
        
def get_subj_ids(subj_files):
    """
    Compiles a list of unique subject id numbers
    Ex: UPENN-GBM-00001 --> 00001

    Args:
        subj_files (list): list of strings of unique subjects

    Returns:
        list of strings
    """
    # get unique subj train ids: [0001, 0002, 0003 ..]
    subj_ids = []
    for subj in subj_files:
        subj_id = mri_common.get_mri_subj_id(subj)
        if subj_id not in subj_ids:
            subj_ids.append(subj_id)
    return subj_ids

def get_subset_files(subj_files, file_no_min, file_no_max, subj_id_min, subj_id_max):
    """
    Finds all associated MRI and mask volumes for a given subject unique id within
    a specified range of files

    Args:
        subj_files (DataHandler): DataHandler instance (train/val/test)
        file_no_min (int): Minimum 2D slice number
        file_no_max (int): Maximum 2D slice number
        subj_id_min (int): Minimum subject index in subject list
        subj_id_max(int): Maximum subject index in subject list

    Returns:
        list of strings, sorted ascending
    """
    subj_ids = get_subj_ids(subj_files)
    # filter files within range
    subj_filenames = []
    for file_name in subj_files:
        subj_id = mri_common.get_mri_subj_id(file_name)
        subj_idx = subj_ids.index(subj_id)
        file_no = mri_common.get_mri_file_no(file_name)
        if file_no >= file_no_min and file_no < file_no_max and subj_idx >= subj_id_min and subj_idx < subj_id_max:

            # example:
            # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
            # result: UPENN-GBM-00006_1.nii.gz
            subj_filename = mri_common.get_mri_slice_file_name(file_name)

            # item has to be unique
            if subj_filename not in subj_filenames:
                subj_filenames.append(subj_filename)

    subj_sorted = sorted(subj_filenames, key=_extract_numeric_part)
    return subj_sorted

def get_all_subj_ids(data_dir):
    """
    Compiles a list of all unique subject ids

    Ex: files: UPENN-GBM-00006_11_FLAIR_1.nii.gz, UPENN-GBM-00006_11_T1_1.nii.gz ...
    --> unique subj_id: UPENN-GBM-00006
    --> all subject ids [UPENN-GBM-00006, UPENN-GBM-00008, UPENN-GBM-00013, ...]

    Args:
        data_dir (str): path to data directory

    Returns:
        list of strings, sorted ascending
    """
    # Obtain full  dataset
    # UPENN-GBM-00006_11_FLAIR_1.nii.gz, UPENN-GBM-00006_11_T1_1.nii.gz, UPENN-GBM-00006_11_FLAIR_2.nii.gz...
    all_slices = os.listdir(data_dir)

    # get unique subj_ids: UPENN-GBM-00008, UPENN-GBM-00013 ...
    all_subjs = [mri_common.get_mri_subj(slice_2d) for slice_2d in all_slices]
    all_subjs = sorted(list(set(all_subjs)))
    return all_subjs

def plot_mask_labels(class_labels, pixel_values, mask_labels, title, scale=True):
    """
    Creates a figure visualizing an input image slice and its corresponding
    class label masks.

    Args:
        class_labels (list): class labels
        pixel_values (numpy.ndarray): 2D MRI image
        mask_labels (array-like): 2D mask array
        title (str): figure title
        scale (bool): flag indicating whether to binarize and re-scale mask [0, 255]

    Returns:
        None.
    """
    n_image = len(class_labels) + 1
    # plot input pixel values
    fig, axs = plt.subplots(nrows=1, ncols=n_image, figsize=(5*n_image, 5))
    denormalized_img = denormalize_img(pixel_values)

    mri_plt.plot_img(img_data=denormalized_img, title=title, fig=fig, axs=axs, row=0, col=0, colorbar=False)

    # plot mask for each label
    for mask_idx, mask_id in enumerate(class_labels):
        visual_mask = scale_mask(mask_labels[mask_idx]) if scale else mask_labels[mask_idx]
        segment_name = mri_common.SEGMENTS[mask_id]
        mri_plt.plot_img(img_data=visual_mask, title=segment_name, fig=fig, axs=axs, row=0, col=mask_idx+1, colorbar=False)

    plt.tight_layout()
    plt.show()

def plot_mask_comparison(input_class_labels, pred_class_labels, input_pixel_values, input_mask_labels, pred_mask_labels, title, scale=True):
    """
    Plots predicted segment over input of segment
    note: if a segment is predicted but does not exist in the input mask,
    it will not be visualized

    Args:
        input_class_labels (numpy array-like): ground truth labels
        pred_class_labels (numpy array-like): predicted labels
        input_pixel_values (numpy array-like): 2D input image
        input_mask_labels (numpy array-like): 2D ground truth mask
        pred_mask_labels (numpy array-like): 2D predicted mask
        title (str): figure title
        scale (bool): flag indicating whether to binarize mask and scale [0, 255]

    Returns:
        None.
    """

    n_image = len(input_class_labels) + 1
    fig, axs = plt.subplots(nrows=1, ncols=n_image, figsize=(5*n_image, 5))
    # plot input image
    denormalized_img = denormalize_img(input_pixel_values)
    mri_plt.plot_img(img_data=denormalized_img, title=title, fig=fig, axs=axs, row=0, col=0)

    # plot binary mask for each segment
    pred_mask_idx = 0
    for true_mask_idx, label_id in enumerate(input_class_labels):
        axs_idx = true_mask_idx + 1
        segment_name = mri_common.SEGMENTS[label_id]
        input_visual_mask = scale_mask(input_mask_labels[true_mask_idx]) if scale else input_mask_labels[true_mask_idx]

        if label_id in pred_class_labels:
            pred_visual_mask = scale_mask(pred_mask_labels[pred_mask_idx]) if scale else pred_mask_labels[pred_mask_idx]
            pred_mask_idx+=1
        else:
            pred_visual_mask =  np.zeros(input_visual_mask.shape, dtype=np.uint8)

        mri_plt.plot_masks(masks=[input_visual_mask, pred_visual_mask],
                           legends=["input", "predicted"],
                           title=segment_name,
                           fig=fig, axs=axs, row=0, col=axs_idx)

    plt.tight_layout()
    plt.show()

def plot_segmentation_comparison(input_pixel_values, input_segmentation, pred_segmentation, title, loc="lower right"):
    """
    Plots ground truth and predicted segmentation masks as single image instead of separately.

    Args:
        input_pixel_values (array-like): 2D MRI image
        input_segmentation (array-like): 2D ground truth segmentation mask
        pred_segmentation (array-like): 2D predicted segmentation mask
        title (str): Figure title
        loc (str): location for placing axes legend ["lower right", "upper right", "lower left", "upper left"]

    Returns:
        None
    """
    n_image = 3
    fig, axs = plt.subplots(nrows=1, ncols=n_image, figsize=(4*n_image, 4))
    denormalized_img = denormalize_img(input_pixel_values)
    mri_plt.plot_img(img_data=denormalized_img, title=title, fig=fig, axs=axs, row=0, col=0)
    mri_plt.plot_segm_img(img_data=input_segmentation, fig=fig, axs=axs, row=0, col=1, use_legend=True, loc=loc)
    mri_plt.plot_segm_img(img_data=pred_segmentation, fig=fig, axs=axs, row=0, col=2, use_legend=True, loc=loc)

    plt.tight_layout()
    plt.show()

def resize_mask(mask, original_size):
    """
    Rescale mask to original dimensions

    Args:
        mask (array-like): segmentation mask
        original_size (tuple): tuple of ints representing original image dimensions

    Returns:
        array-like
    """
    # Note: cv2.resize expects the size in (width, height) format
    original_size = (original_size[1], original_size[0])  # Switching from HxW to WxH

    # Check if mask has multiple channels
    if len(mask.shape) > 2:
        # Create a list to store each resized channel
        resized_mask_channels = []

        # Loop over each channel in the mask
        for channel in mask:
            # Resize the channel and append to the list
            resized_mask_channel = cv2.resize(channel, original_size, interpolation=cv2.INTER_NEAREST)
            resized_mask_channels.append(resized_mask_channel)

        # Stack the resized channels back into a single mask
        resized_mask = np.stack(resized_mask_channels, axis=0)
    else:
        # If mask has only one channel, just resize it
        resized_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    return resized_mask

def _extract_numeric_part(filename):
    """
    Helper function for extracting slice number for sorting
    Ex:
    input filename: 'UPENN-GBM-00131_50.nii.gz'
    prefix: 'UPENN-GBM-00131'
    numeric_part: 50

    Args:
        filename (str): a valid file name with a unique subj id and a slice number
    """
    # Expected filename: "UPENN-GBM-00131_50.nii.gz"
    # Split the filename based on underscore

    parts = filename.split('_')
    if len(parts) == 2:
        prefix = parts[0]
        numeric_part = int(parts[1].split('.')[0])
        return prefix, numeric_part
    return filename, -1  # Return the original filename if the format is not as expected

