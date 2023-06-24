import medpy.metric.binary as mdp

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support

# region Commonly used metrics
def calc_binary_metrics(segm_pred, segm_true, segm_id=None):
    '''
    precision: tp / tp + fp => positive predicted value
    recall: tp / tp + fn => ratio of true positives to total positives in the data
    specificity = tn / (tn + fp) => ratio of true nagatives to total negatives in the data
    '''
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)

    recall = mdp.recall(mask_pred, mask_true)
    specificity = mdp.specificity(mask_pred, mask_true)
    metrics_dict = {
        "true_positive": np.count_nonzero(mask_pred & mask_true),
        "true_negative": np.count_nonzero(~mask_pred & ~mask_true),
        "false_positive": np.count_nonzero(mask_pred & ~mask_true),
        "false_negative": np.count_nonzero(~mask_pred & mask_true),
        "precision": mdp.precision(mask_pred, mask_true),
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
        "true_positve_rate": recall,
        "true_negative_rate": mdp.true_negative_rate(mask_pred, mask_true)
    } 
    return metrics_dict

def calc_dice_score(segm_pred, segm_true, segm_id=None):
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)

    # perform element wise multiplication to get intersection
    intersection = np.sum(mask_pred * mask_true)
    dice_score = (2.0 * intersection) / (np.sum(mask_true) + np.sum(mask_pred))
    return dice_score
  
def calc_miou(segm_pred, segm_true, segm_id=None):
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)
        
    num_i = len( np.where(mask_pred * mask_true != 0)[0] )
    num_u = len( np.where(mask_pred + mask_true != 0)[0] )
    miou = num_i / num_u * 100
    return miou, num_u
  
# endregion

# region Boundary-based metrics
def calc_hausdorff_95(segm_pred, segm_true, segm_id=None):
    return _exec_hausdorff_95(segm_pred, segm_true, segm_id=segm_id, plot=False)

def plot_hausdorff_95(segm_pred, segm_true, segm_id=None):
    return _exec_hausdorff_95(segm_pred, segm_true, segm_id=segm_id, plot=True)

def _calc_surface_distances(mask_pred, true_pred, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.

    Source Code: https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L357
    """
    mask_pred = np.atleast_1d(mask_pred.astype(np.bool))
    true_pred = np.atleast_1d(true_pred.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, mask_pred.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # this will create 3x3 binary array with False values at the center
    # and its surrounding structures. diagonals are not considered if connectivity = 1
    # binary structure
    footprint = generate_binary_structure(mask_pred.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(mask_pred): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(true_pred): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
    
    # extract only 1-pixel border line of objects
    # applies binary erosion to the image
    pred_border = mask_pred ^ binary_erosion(mask_pred, structure=footprint, iterations=1)
    true_border = true_pred ^ binary_erosion(true_pred, structure=footprint, iterations=1)
    
    # calculate surface distance of each pixel 
    dist = distance_transform_edt(~true_border, sampling=voxelspacing)

    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    surface_dist = np.where(pred_border, dist, -1)
    return surface_dist


def _exec_hausdorff_95(segm_pred, segm_true, segm_id=None, plot=False):
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)
        
    if np.sum(mask_true) + np.sum(mask_pred) == 0:
        print("Both images do not have the segment ids")
        return None
    
    connectivity=1
    if mask_true.ndim == 3:
        connectivity = 26
    elif mask_true.ndim == 2:
        connectivity = 8
         
    if plot:
        all_dist_pred = _calc_surface_distances(mask_pred, mask_true, connectivity=connectivity)
        all_dist_true = _calc_surface_distances(mask_true, mask_pred, connectivity=connectivity)
        surface_dist_pred = all_dist_pred[all_dist_pred >= 0]
        surface_dist_true = all_dist_true[all_dist_true >= 0]
        
        hd95_val = np.percentile(np.hstack((surface_dist_true, surface_dist_pred)), 95)
        return hd95_val, surface_dist_pred, surface_dist_true
    else:
        return mdp.hd95(mask_pred, mask_true, connectivity=connectivity)

# endregion

# region Helper Functions
def _get_binary_mask(img, segm_id):
    return img == segm_id
# endregion