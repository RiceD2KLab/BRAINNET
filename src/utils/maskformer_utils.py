'''Common functions used by maskformer (segmentation)'''
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# local imports
import utils.mri_common as mri_common

from utils.mri_plotter import MRIPlotter
from utils.data_handler import DataHandler, MriType

mri_plt = MRIPlotter()

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

def scale_mask(mask):
    if isinstance(mask, torch.Tensor):
        visual_mask = (mask.bool().numpy() * 255).astype(np.uint8)
    else:
        visual_mask = (mask * 255).astype(np.uint8)
    return visual_mask
    # return Image.fromarray(visual_mask)

def descale_mask(mask):
    return (mask/255).astype(np.uint8)

def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  return scale_mask(mask)

def softmax(logits):
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)

def softmax_2d(logits):
    e_x = np.exp(logits - np.max(logits, axis=(0, 1), keepdims=True))
    return e_x / np.sum(e_x, axis=(0, 1), keepdims=True)

def denormalize_img(img_data):
    denormalized_image = ( img_data * np.array(ADE_STD)[:, None, None] ) + np.array(ADE_MEAN)[:, None, None]
    denormalized_image = ( denormalized_image * 255 ).astype(np.uint8)
    denormalized_image = np.moveaxis(denormalized_image, 0, -1)
    return denormalized_image

def to_brats_mask(mask_3d):
    # assumed shape:  (4, 146, 512, 512)
    # init brats mask
    brats_mask_shape = list(mask_3d.shape)
    brats_mask_shape[0] = len(mri_common.BRATS_REGIONS)
    brats_mask =  np.zeros(tuple(brats_mask_shape), dtype=np.uint8)
    
    for brats_region_idx, region_mapping in enumerate(mri_common.BRATS_REGIONS.items()):
        prev_brats_region = brats_mask[brats_region_idx, :, :, :]
        for sub_region in region_mapping[1]:
            sub_region_mask = mask_3d[sub_region, :, :, :]
            prev_brats_region = np.logical_or(prev_brats_region, sub_region_mask)

        brats_mask[brats_region_idx, :, :, :] = prev_brats_region
    return brats_mask

def input_mask_to_segmentation(input_mask_labels, class_labels):
    # map (4,512,512) to segmentation (512, 512)
    # or map(4, 146, 512, 512) to (146, 512, 412)
    input_mask_mapping = {label_id: label_id for label_id in class_labels}
    input_mask_to_segm = np.vectorize(input_mask_mapping.get)(input_mask_labels.argmax(axis=0))
    return input_mask_to_segm.astype(np.uint8)

def post_proc_result_to_segmentation(results):
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

def get_subj_ids(subj_files):
    # get unique subj train ids: [0001, 0002, 0003 ..]
    subj_ids = []
    for subj in subj_files:
        subj_id = mri_common.get_mri_subj_id(subj)
        if subj_id not in subj_ids:
            subj_ids.append(subj_id)
    return subj_ids

def get_subset_files(subj_files, file_no_min, file_no_max, subj_id_min, subj_id_max):
    
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
    subj_filenames.sort()
    return subj_filenames

def get_all_subj_ids(data_handler: DataHandler, mri_type: MriType):
    # Obtain full  dataset
    # UPENN-GBM-00006_11_FLAIR_1.nii.gz, UPENN-GBM-00006_11_T1_1.nii.gz, UPENN-GBM-00006_11_FLAIR_2.nii.gz...
    all_slices = data_handler.list_mri_in_dir(mri_type=mri_type)

    # get unique subj_ids: UPENN-GBM-00008, UPENN-GBM-00013 ...
    all_subjs = [mri_common.get_mri_subj(slice_2d) for slice_2d in all_slices]
    all_subjs = sorted(list(set(all_subjs)))
    return all_subjs

def plot_mask_labels(class_labels, pixel_values, mask_labels, title, scale=True):
    n_image = len(class_labels) + 1
    # plot input pixel values
    fig, axs = plt.subplots(nrows=1, ncols=n_image, figsize=(5*n_image, 5))
    denormalized_img = denormalize_img(pixel_values)

    mri_plt.plot_img(img_data=denormalized_img, title=title, fig=fig, axs=axs, row=0, col=0, colorbar=True)

    # plot mask for each label
    for mask_idx, mask_id in enumerate(class_labels):
        visual_mask = scale_mask(mask_labels[mask_idx]) if scale else mask_labels[mask_idx]
        segment_name = mri_common.SEGMENTS[mask_id]
        mri_plt.plot_img(img_data=visual_mask, title=segment_name, fig=fig, axs=axs, row=0, col=mask_idx+1, colorbar=True)

    plt.tight_layout()
    plt.show()

def plot_mask_comparison(input_class_labels, pred_class_labels, input_pixel_values, input_mask_labels, pred_mask_labels, title, scale=True):

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

def plot_segmentation_comparison(input_pixel_values, input_segmentation, pred_segmentation, title):
    n_image = 3
    fig, axs = plt.subplots(nrows=1, ncols=n_image, figsize=(4*n_image, 4))
    denormalized_img = denormalize_img(input_pixel_values)
    mri_plt.plot_img(img_data=denormalized_img, title=title, fig=fig, axs=axs, row=0, col=0)

    mri_plt.plot_segm_img(img_data=input_segmentation, fig=fig, axs=axs, row=0, col=1, use_legend=True)
    mri_plt.plot_segm_img(img_data=pred_segmentation, fig=fig, axs=axs, row=0, col=2, use_legend=True)

    plt.tight_layout()
    plt.show()

def resize_mask(mask, original_size):
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