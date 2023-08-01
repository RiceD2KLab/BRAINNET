'''Metrics for both classification and segmentation'''

from enum import Enum
import medpy.metric.binary as mdp
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support

# region Commonly used metrics
class MetricName(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    DICE_SCORE = "dice_score"
    HD95 = "hd95"
    
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
        MetricName.PRECISION.value : mdp.precision(mask_pred, mask_true),
        MetricName.RECALL.value: recall,
        MetricName.SPECIFICITY.value: specificity,
        MetricName.SENSITIVITY.value: recall
    } 
    return metrics_dict

def calc_dice_score(segm_pred, segm_true, segm_id=None):
    mask_true = segm_true
    mask_pred = segm_pred
    
    # handling if we forgot to scale back image from 0 to 255 to 0 to 1
    if mask_true.max() > 1:
        mask_true = np.clip(mask_true, 0, 1)  
    if mask_pred.max() > 1:
        mask_pred = np.clip(mask_pred, 0, 1)  
     
    # segm_id can be passed to convert the segmented image to binary mask if not yet converted
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)

    # test for emptiness
    if (np.sum(mask_true) + np.sum(mask_pred)) == 0:
        msg = "this segment." if segm_id is None else f"segment {segm_id}."
        print(f"Both images do not have {msg}")
        return 1
    
    return mdp.dc(mask_pred, mask_true)
# endregion

# region Boundary-based metrics
def calc_hausdorff_95(segm_pred, segm_true, segm_id=None, connectivity=1):
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)
        
    if np.sum(mask_true) + np.sum(mask_pred) == 0:
        msg = "this segment." if segm_id is None else f"segment {segm_id}."
        print(f"Both images do not have {msg}")
        return 0    
    try:
        # print("connectivity", connectivity)
        # connectivity=1
        # if mask_true.ndim == 3:
        #     connectivity = 26
        # elif mask_true.ndim == 2:
        #     connectivity = 8
        
        # KAIST 2021 BraTs winner does not have connectivity param in their code
        return mdp.hd95(mask_pred, mask_true, connectivity=connectivity)
    except:
        return np.inf

# endregion

# region Helper Functions
def _get_binary_mask(img, segm_id):
    return img == segm_id
# endregion